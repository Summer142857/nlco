
import os
os.environ["GRB_LICENSE_FILE"] = r"C:\Users\20240908\Desktop\mygits\co_benchmark\ss\gurobi.lic"

import xml.etree.ElementTree as ET
import argparse
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

def parse_ros_instance(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # helper to find text safely
    def get_text(elem, path):
        found = elem.find(path)
        return found.text if found is not None else None

    # Dates
    start_date_str = get_text(root, 'StartDate')
    end_date_str = get_text(root, 'EndDate')
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    num_days = (end_date - start_date).days + 1
    
    # Shifts
    shifts = {}
    for shift in root.find('ShiftTypes'):
        sid = shift.get('ID')
        duration = int(get_text(shift, 'Duration'))
        start_time_str = get_text(shift, 'StartTime')
        start_mins = 0
        if start_time_str:
            h, m = map(int, start_time_str.split(':'))
            start_mins = h * 60 + m
            
        shifts[sid] = {'duration': duration, 'start_time': start_mins}
    
    # Contracts
    # We parse all contracts into a dictionary
    contracts_raw = {}
    for contract in root.find('Contracts'):
        cid = contract.get('ID')
        c_data = {}
        
        # MinRestTime
        mrt = get_text(contract, 'MinRestTime')
        if mrt is not None:
             c_data['min_rest_time'] = int(mrt)
        
        # Workload
        workload = contract.find('Workload')
        if workload is not None:
             time_units = workload.findall('TimeUnits')
             for tu in time_units:
                 mx = tu.find('Max')
                 mn = tu.find('Min')
                 if mx is not None: c_data['max_minutes'] = int(get_text(mx, 'Count'))
                 if mn is not None: c_data['min_minutes'] = int(get_text(mn, 'Count'))
        
        # MaxSeq
        for max_seq in contract.findall('MaxSeq'):
            if max_seq.get('shift') == '$':
                c_data['max_consecutive_shifts'] = int(max_seq.get('value'))
        
        # MinSeq
        for min_seq in contract.findall('MinSeq'):
            val = int(min_seq.get('value'))
            if min_seq.get('shift') == '$':
                 c_data['min_consecutive_shifts'] = val
            elif min_seq.get('shift') == '-':
                 c_data['min_consecutive_days_off'] = val

        # Patterns
        p_match = contract.find('Patterns/Match')
        if p_match is not None:
            mx = p_match.find('Max') 
            if mx is not None:
                 label = get_text(mx, 'Label')
                 if label and 'weekend' in label.lower():
                     c_data['max_weekends'] = int(get_text(mx, 'Count'))
        
        # ValidShifts
        vs_tag = contract.find('ValidShifts')
        if vs_tag is not None:
            vs_str = vs_tag.get('shift')
            c_data['valid_shifts_raw'] = vs_str

        contracts_raw[cid] = c_data

    # Employees & Contract Resolution
    employees = []
    # Base defaults for a contract
    default_contract = {
        'min_minutes': 0, 'max_minutes': 999999,
        'max_consecutive_shifts': 999, 'min_consecutive_shifts': 0,
        'min_consecutive_days_off': 0, 'max_weekends': 999,
        'min_rest_time': 0,
        'valid_shifts': set(shifts.keys()) # Default all valid
    }

    for emp in root.find('Employees'):
        eid = emp.get('ID')
        cids = [c.text for c in emp.findall('ContractID')]
        
        # Merge contracts in order
        merged = default_contract.copy()
        
        # If 'All' is implicitly a base, we start with it if present, but the XML logic implies inheritance order in list.
        # But 'valid_shifts' logic: if a contract specifies it, it Restricts or Resets?
        # Usually checking benchmarks: "The set of valid shifts for an employee is the INTERSECTION of valid sets?"
        # Or usually just the LAST non-empty definition wins? 
        # In Instance3.ros, 'All' doesn't specify ValidShifts. 'A' specifies E,D. So A can only work E,D.
        # So we can assume we start with ALL shifts valid, and if any contract specifies a subset, we intersect.
        
        current_valid = set(shifts.keys())

        for cid in cids:
            if cid in contracts_raw:
                raw = contracts_raw[cid]
                # Update scalar fields (overwrite)
                for k in ['min_minutes', 'max_minutes', 'max_consecutive_shifts', 
                          'min_consecutive_shifts', 'min_consecutive_days_off', 
                          'max_weekends', 'min_rest_time']:
                    if k in raw:
                        merged[k] = raw[k]
                
                # Valid Shifts
                if 'valid_shifts_raw' in raw:
                    vs_str = raw['valid_shifts_raw']
                    vs = [s for s in vs_str.split(',') if s.strip()]
                    current_valid = current_valid.intersection(vs)
        
        merged['valid_shifts'] = current_valid
        employees.append({'id': eid, 'contract_data': merged})

    # Fixed Assignments
    fixed = {} # (emp_id, day_idx) -> shift_id
    if root.find('FixedAssignments') is not None:
         for fa in root.find('FixedAssignments').findall('Employee'):
             emp_id = get_text(fa, 'EmployeeID')
             assign = fa.find('Assign')
             if assign is not None:
                 day = int(get_text(assign, 'Day'))
                 shift = get_text(assign, 'Shift')
                 fixed[(emp_id, day)] = shift

    # Requests
    shift_off = []
    if root.find('ShiftOffRequests') is not None:
        for req in root.find('ShiftOffRequests'):
            shift_off.append({
                'employee': get_text(req, 'EmployeeID'),
                'day': int(get_text(req, 'Day')),
                'shift': get_text(req, 'Shift'),
                'weight': int(req.get('weight'))
            })
            
    shift_on = []
    if root.find('ShiftOnRequests') is not None:
        for req in root.find('ShiftOnRequests'):
            shift_on.append({
                'employee': get_text(req, 'EmployeeID'),
                'day': int(get_text(req, 'Day')),
                'shift': get_text(req, 'Shift'),
                'weight': int(req.get('weight'))
            })

    # Cover Requirements
    cover = {} # (day, shift) -> {min: (val, w), max: (val, w), pref: (val, w)}
    if root.find('CoverRequirements') is not None:
        for dsc in root.find('CoverRequirements').findall('DateSpecificCover'):
            day = int(get_text(dsc, 'Day'))
            for cv in dsc.findall('Cover'):
                shift = get_text(cv, 'Shift')
                
                mn = cv.find('Min')
                mx = cv.find('Max')
                pref = cv.find('Preferred')
                
                c_data = {}
                if mn is not None: c_data['min'] = (int(mn.text), int(mn.get('weight')))
                if mx is not None: c_data['max'] = (int(mx.text), int(mx.get('weight')))
                if pref is not None: c_data['pref'] = (int(pref.text), int(pref.get('weight')))
                
                cover[(day, shift)] = c_data

    return {
        'num_days': num_days,
        'start_date': start_date,
        'sub_shifts': shifts,
        'employees': employees,
        'fixed': fixed,
        'shift_off': shift_off,
        'shift_on': shift_on,
        'cover': cover
    }

def solve_nrp(instance_data: Dict[str, Any], time_limit: float = 60, verbose: bool = False) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    model = gp.Model("NRP")
    model.setParam('TimeLimit', time_limit)
    
    # Data
    num_days = instance_data['num_days']
    days = range(num_days)
    employees = instance_data['employees']
    shifts_dict = instance_data['sub_shifts']
    shift_ids = list(shifts_dict.keys())
    
    # Helper: Weekend identification
    weekends = []
    for d in range(num_days):
        if d % 7 == 5: # Saturday
            if d + 1 < num_days:
                weekends.append((d, d+1))

    # Variables
    x = {}
    k = {}
    
    for e in employees:
        eid = e['id']
        contract = e['contract_data']
        valid_shifts = contract['valid_shifts']
        
        for d in days:
            k[eid, d] = model.addVar(vtype=GRB.BINARY, name=f"k_{eid}_{d}")
            
            for s in shift_ids:
                if s in valid_shifts:
                    x[eid, d, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{eid}_{d}_{s}")
                else:
                    # Invalid shift
                    x[eid, d, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{eid}_{d}_{s}")
                    model.addConstr(x[eid, d, s] == 0, name=f"invalid_{eid}_{d}_{s}")
            
            # Linking k and x
            model.addConstr(k[eid, d] == gp.quicksum(x[eid, d, s] for s in shift_ids), name=f"link_k_x_{eid}_{d}")
    
    # --- Hard Constraints ---
    for e in employees:
        eid = e['id']
        contract = e['contract_data']
        
        # 1. Workload
        total_mins = gp.quicksum(x[eid, d, s] * shifts_dict[s]['duration'] 
                                 for d in days for s in shift_ids)
        model.addConstr(total_mins >= contract['min_minutes'], name=f"min_load_{eid}")
        model.addConstr(total_mins <= contract['max_minutes'], name=f"max_load_{eid}")
        
        # 2. Max Consecutive Shifts
        max_cons_shifts = contract['max_consecutive_shifts']
        for d in range(num_days - max_cons_shifts):
            model.addConstr(gp.quicksum(k[eid, d+j] for j in range(max_cons_shifts + 1)) <= max_cons_shifts,
                            name=f"max_cons_shifts_{eid}_{d}")
                            
        # 3. Min Consecutive Shifts
        min_cons_shifts = contract['min_consecutive_shifts']
        if min_cons_shifts > 1:
            for d in range(1, num_days - 1):
                if d + min_cons_shifts <= num_days:
                     # Start of block: k[d]=1, k[d-1]=0
                     start_work = model.addVar(vtype=GRB.BINARY)
                     model.addConstr(start_work >= k[eid, d] - k[eid, d-1])
                     for j in range(1, min_cons_shifts):
                         model.addConstr(k[eid, d+j] >= start_work)
            
            # Boundary (Day 0)
            start_0 = k[eid, 0]
            if min_cons_shifts > 1 and num_days >= min_cons_shifts:
                 for j in range(1, min_cons_shifts):
                     model.addConstr(k[eid, j] >= start_0)

        # 4. Min Consecutive Days Off
        min_cons_off = contract['min_consecutive_days_off']
        if min_cons_off > 1:
            for d in range(1, num_days - 1):
                if d + min_cons_off <= num_days:
                     # Start of off block: k[d-1]=1, k[d]=0
                     start_off = model.addVar(vtype=GRB.BINARY)
                     model.addConstr(start_off >= k[eid, d-1] - k[eid, d])
                     for j in range(1, min_cons_off):
                         model.addConstr(k[eid, d+j] <= 1 - start_off)
        
        # 5. Max Weekends
        max_weekends = contract['max_weekends']
        weekend_vars = []
        for (sat, sun) in weekends:
            w_worked = model.addVar(vtype=GRB.BINARY, name=f"weekend_worked_{eid}_{sat}")
            model.addConstr(w_worked >= k[eid, sat])
            model.addConstr(w_worked >= k[eid, sun])
            weekend_vars.append(w_worked)
        model.addConstr(gp.quicksum(weekend_vars) <= max_weekends, name=f"max_weekends_{eid}")

        # 6. Shift Rotation (New)
        min_rest = contract['min_rest_time']
        # Identify forbidden pairs (s1 -> s2 next day)
        forbidden_pairs = []
        for s1 in shift_ids:
            for s2 in shift_ids:
                end_s1 = shifts_dict[s1]['start_time'] + shifts_dict[s1]['duration']
                start_s2_next = shifts_dict[s2]['start_time'] + 1440 # Next day
                if end_s1 + min_rest > start_s2_next:
                    forbidden_pairs.append((s1, s2))
        
        if forbidden_pairs:
            for d in range(num_days - 1):
                for (s1, s2) in forbidden_pairs:
                    model.addConstr(x[eid, d, s1] + x[eid, d+1, s2] <= 1, 
                                    name=f"rotation_{eid}_{d}_{s1}_{s2}")

    # --- Soft Constraints (Objective) ---
    obj_terms = []
    
    # 1. Cover Requirements
    # Note: If weights are large (like 100), effectively hard-ish soft constraints.
    for d in days:
        for s in shift_ids:
            if (d, s) in instance_data['cover']:
                req = instance_data['cover'][(d, s)]
                staffing = gp.quicksum(x[e['id'], d, s] for e in employees)
                
                # Min
                if 'min' in req:
                    target, weight = req['min']
                    if weight > 0:
                        under = model.addVar(lb=0, name=f"under_{d}_{s}")
                        model.addConstr(under >= target - staffing)
                        obj_terms.append(weight * under)
                
                # Max
                if 'max' in req:
                    target, weight = req['max']
                    if weight > 0:
                        over = model.addVar(lb=0, name=f"over_{d}_{s}")
                        model.addConstr(over >= staffing - target)
                        obj_terms.append(weight * over)

    # 2. Requests
    for req in instance_data['shift_off']:
        eid = req['employee']
        d = req['day']
        s = req['shift']
        w = req['weight']
        if s in shift_ids:
            obj_terms.append(w * x[eid, d, s])

    for req in instance_data['shift_on']:
        eid = req['employee']
        d = req['day']
        s = req['shift']
        w = req['weight']
        if s in shift_ids:
            obj_terms.append(w * (1 - x[eid, d, s]))
    
    # 3. Fixed Assignments (Hard)
    for (eid, d), s in instance_data['fixed'].items():
        if s in shift_ids:
             # If assigned shift is invalid for employee, this writes an infeasible constraint unless we check.
             # But benchmarks usually assume valid fixed assignments. 
             model.addConstr(x[eid, d, s] == 1, name=f"fixed_{eid}_{d}_{s}")
        elif s == '-': 
             model.addConstr(k[eid, d] == 0, name=f"fixed_off_{eid}_{d}")

    model.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
    model.optimize()

    if model.SolCount <= 0:
        if verbose:
            print("No solution found.")
        return None, None

    objective_value = float(model.ObjVal)

    schedule_by_employee: Dict[str, List[str]] = {e['id']: ["-"] * num_days for e in employees}
    for d in days:
        for e in employees:
            eid = e['id']
            assigned = "-"
            val_k = k[eid, d].X
            if val_k > 0.5:
                for s in shift_ids:
                    if x[eid, d, s].X > 0.5:
                        assigned = s
                        break
            schedule_by_employee[eid][d] = assigned

    solution: Dict[str, Any] = {
        "employee_ids": [e["id"] for e in employees],
        "num_days": int(num_days),
        "schedule": schedule_by_employee,
    }

    if verbose:
        print(f"Objective Value: {objective_value}")
        print("Schedule:")
        print("Day\t" + "\t".join([e['id'] for e in employees]))
        for d in days:
            row = [str(d)]
            for e in employees:
                row.append(schedule_by_employee[e['id']][d])
            print("\t".join(row))

    return objective_value, solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve NRP instance.')
    parser.add_argument('--instance', type=str, default=r"C:\Users\20240908\Desktop\mygits\co_benchmark\step1_instance_creation\datasets\NRP\Instance5.ros", help='Path to .ros instance file')
    args = parser.parse_args()
    
    data = parse_ros_instance(args.instance)
    obj, _sol = solve_nrp(data, verbose=True)
    if obj is None:
        raise SystemExit(1)
