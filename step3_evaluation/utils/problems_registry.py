
from typing import Any, Dict, Callable, Optional, Literal
from step3_evaluation.problem.eval_cvrp import cvrp_parse_instance, cvrp_check_feasibility, cvrp_objective_model
from step3_evaluation.problem.eval_csp import csp_parse_instance, csp_check_feasibility, csp_objective_model
from step3_evaluation.problem.eval_gcp import gcp_parse_instance, gcp_objective_model, gcp_check_feasibility
from step3_evaluation.problem.eval_maxcut import maxcut_parse_instance, maxcut_objective_model, maxcut_check_feasibility
from step3_evaluation.problem.eval_mcp import mcp_parse_instance, mcp_check_feasibility, mcp_objective_model
from step3_evaluation.problem.eval_mds import mds_parse_instance, mds_check_feasibility, mds_objective_model
from step3_evaluation.problem.eval_mlp import mlp_parse_instance, mlp_check_feasibility, mlp_objective_model
from step3_evaluation.problem.eval_mvc import mvc_parse_instance, mvc_objective_model, mvc_check_feasibility
from step3_evaluation.problem.eval_op import op_parse_instance, op_check_feasibility, op_objective_model
from step3_evaluation.problem.eval_pctsp import pctsp_parse_instance, pctsp_check_feasibility, pctsp_objective_model
from step3_evaluation.problem.eval_pdp import pdp_parse_instance, pdp_check_feasibility, pdp_objective_model
from step3_evaluation.problem.eval_tsp import tsp_parse_instance, tsp_check_feasibility, tsp_objective_model
from step3_evaluation.problem.eval_tsptw import tsptw_parse_instance, tsptw_objective_model, tsptw_check_feasibility
from step3_evaluation.problem.eval_gap import gap_parse_instance, gap_check_feasibility, gap_objective_model
from step3_evaluation.problem.eval_kp import kp_parse_instance, kp_check_feasibility, kp_objective_model
from step3_evaluation.problem.eval_qkp import qkp_parse_instance, qkp_check_feasibility, qkp_objective_model
from step3_evaluation.problem.eval_mkc import mkc_parse_instance, mkc_check_feasibility, mkc_objective_model
from step3_evaluation.problem.eval_hsp import hsp_parse_instance, hsp_check_feasibility, hsp_objective_model
from step3_evaluation.problem.eval_spp import spp_parse_instance, spp_check_feasibility, spp_objective_model
from step3_evaluation.problem.eval_sp import sp_parse_instance, sp_check_feasibility, sp_objective_model
from step3_evaluation.problem.eval_scp import scp_parse_instance, scp_check_feasibility, scp_objective_model
from step3_evaluation.problem.eval_uflp import uflp_parse_instance, uflp_check_feasibility, uflp_objective_model
from step3_evaluation.problem.eval_smtwt import smtwt_parse_instance, smtwt_check_feasibility, smtwt_objective_model
from step3_evaluation.problem.eval_stp import stp_parse_instance, stp_check_feasibility, stp_objective_model
from step3_evaluation.problem.eval_sfp import sfp_parse_instance, sfp_check_feasibility, sfp_objective_model
from step3_evaluation.problem.eval_rcpsp import rcpsp_parse_instance, rcpsp_check_feasibility, rcpsp_objective_model
from step3_evaluation.problem.eval_qap import qap_parse_instance, qap_check_feasibility, qap_objective_model
from step3_evaluation.problem.eval_pmed import pmed_parse_instance, pmed_check_feasibility, pmed_objective_model
from step3_evaluation.problem.eval_pms import pms_parse_instance, pms_check_feasibility, pms_objective_model
from step3_evaluation.problem.eval_pcenter import pcenter_parse_instance, pcenter_check_feasibility, pcenter_objective_model
from step3_evaluation.problem.eval_osp import osp_parse_instance, osp_check_feasibility, osp_objective_model
from step3_evaluation.problem.eval_kmst import kmst_parse_instance, kmst_check_feasibility, kmst_objective_model
from step3_evaluation.problem.eval_jsp import jssp_parse_instance, jssp_check_feasibility, jssp_objective_model
from step3_evaluation.problem.eval_fsp import fsp_parse_instance, fsp_check_feasibility, fsp_objective_model
from step3_evaluation.problem.eval_cmp import cmp_parse_instance, cmp_check_feasibility, cmp_objective_model
from step3_evaluation.problem.eval_cflp import cflp_parse_instance, cflp_check_feasibility, cflp_objective_model
from step3_evaluation.problem.eval_bpp import bpp_parse_instance, bpp_check_feasibility, bpp_objective_model
from step3_evaluation.problem.eval_ap3 import ap3_parse_instance, ap3_check_feasibility, ap3_objective_model
from step3_evaluation.problem.eval_qspp import qspp_parse_instance, qspp_check_feasibility, qspp_objective_model
from step3_evaluation.problem.eval_lop import lop_parse_instance, lop_check_feasibility, lop_objective_model
from step3_evaluation.problem.eval_2sp import twosp_parse_instance, twosp_check_feasibility, twosp_objective_model
from step3_evaluation.problem.eval_mis import mis_parse_instance, mis_check_feasibility, mis_objective_model
from step3_evaluation.problem.eval_mdp import mdp_parse_instance, mdp_check_feasibility, mdp_objective_model

PROBLEM_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "MIS": {
        "parse_instance_fn": mis_parse_instance,
        "feasibility_fn": mis_check_feasibility,
        "objective_model_fn": mis_objective_model,
    },
    "CVRP": {
        "parse_instance_fn": cvrp_parse_instance,
        "feasibility_fn": cvrp_check_feasibility,
        "objective_model_fn": cvrp_objective_model,
    },
    "MAXCUT": {
        "parse_instance_fn": maxcut_parse_instance,
        "feasibility_fn": maxcut_check_feasibility,
        "objective_model_fn": maxcut_objective_model,
    },
    "TSP": {
        "parse_instance_fn": tsp_parse_instance,
        "feasibility_fn": tsp_check_feasibility,
        "objective_model_fn": tsp_objective_model,
    },
    "TSPTW": {
        "parse_instance_fn": tsptw_parse_instance,
        "feasibility_fn": tsptw_check_feasibility,
        "objective_model_fn": tsptw_objective_model,
    },
    "OP": {
        "parse_instance_fn": op_parse_instance,
        "feasibility_fn": op_check_feasibility,
        "objective_model_fn": op_objective_model,
    },
    "PCTSP": {
        "parse_instance_fn": pctsp_parse_instance,
        "feasibility_fn": pctsp_check_feasibility,
        "objective_model_fn": pctsp_objective_model,
    },
    "MLP": {
        "parse_instance_fn": mlp_parse_instance,
        "feasibility_fn": mlp_check_feasibility,
        "objective_model_fn": mlp_objective_model,
    },
    "MCP": {
        "parse_instance_fn": mcp_parse_instance,
        "feasibility_fn": mcp_check_feasibility,
        "objective_model_fn": mcp_objective_model,
    },
    "MDS": {
        "parse_instance_fn": mds_parse_instance,
        "feasibility_fn": mds_check_feasibility,
        "objective_model_fn": mds_objective_model,
    },
    "GCP": {
        "parse_instance_fn": gcp_parse_instance,
        "feasibility_fn": gcp_check_feasibility,
        "objective_model_fn": gcp_objective_model,
    },
    "MVC": {
    "parse_instance_fn": mvc_parse_instance,
    "feasibility_fn": mvc_check_feasibility,
    "objective_model_fn": mvc_objective_model,
    },
    "PDP": {
        "parse_instance_fn": pdp_parse_instance,
        "feasibility_fn": pdp_check_feasibility,
        "objective_model_fn": pdp_objective_model,
    },
    "GAP": {
        "parse_instance_fn": gap_parse_instance,
        "feasibility_fn": gap_check_feasibility,
        "objective_model_fn": gap_objective_model,
    },
    "KP": {
        "parse_instance_fn": kp_parse_instance,
        "feasibility_fn": kp_check_feasibility,
        "objective_model_fn": kp_objective_model,
    },
    "QKP": {
        "parse_instance_fn": qkp_parse_instance,
        "feasibility_fn": qkp_check_feasibility,
        "objective_model_fn": qkp_objective_model,
    },
    "MkC": {
        "parse_instance_fn": mkc_parse_instance,
        "feasibility_fn": mkc_check_feasibility,
        "objective_model_fn": mkc_objective_model,
    },
    "HSP": {
        "parse_instance_fn": hsp_parse_instance,
        "feasibility_fn": hsp_check_feasibility,
        "objective_model_fn": hsp_objective_model,
    },
    "SPP": {
        "parse_instance_fn": spp_parse_instance,
        "feasibility_fn": spp_check_feasibility,
        "objective_model_fn": spp_objective_model,
    },
    "SP": {
        "parse_instance_fn": sp_parse_instance,
        "feasibility_fn": sp_check_feasibility,
        "objective_model_fn": sp_objective_model,
    },
    "SCP": {
        "parse_instance_fn": scp_parse_instance,
        "feasibility_fn": scp_check_feasibility,
        "objective_model_fn": scp_objective_model,
    },
    "UFLP": {
        "parse_instance_fn": uflp_parse_instance,
        "feasibility_fn": uflp_check_feasibility,
        "objective_model_fn": uflp_objective_model,
    },
    "MDP": {
        "parse_instance_fn": mdp_parse_instance,
        "feasibility_fn": mdp_check_feasibility,
        "objective_model_fn": mdp_objective_model,
    },
    "SMTWT": {
        "parse_instance_fn": smtwt_parse_instance,
        "feasibility_fn": smtwt_check_feasibility,
        "objective_model_fn": smtwt_objective_model,
    },
    "STP": {
        "parse_instance_fn": stp_parse_instance,
        "feasibility_fn": stp_check_feasibility,
        "objective_model_fn": stp_objective_model,
    },
    "SFP": {
        "parse_instance_fn": sfp_parse_instance,
        "feasibility_fn": sfp_check_feasibility,
        "objective_model_fn": sfp_objective_model,
    },
    "RCPSP": {
        "parse_instance_fn": rcpsp_parse_instance,
        "feasibility_fn": rcpsp_check_feasibility,
        "objective_model_fn": rcpsp_objective_model,
    },
    "QAP": {
        "parse_instance_fn": qap_parse_instance,
        "feasibility_fn": qap_check_feasibility,
        "objective_model_fn": qap_objective_model,
    },
    "PMED": {
        "parse_instance_fn": pmed_parse_instance,
        "feasibility_fn": pmed_check_feasibility,
        "objective_model_fn": pmed_objective_model,
    },
    "PMS": {
        "parse_instance_fn": pms_parse_instance,
        "feasibility_fn": pms_check_feasibility,
        "objective_model_fn": pms_objective_model,
    },
    "PCENTER": {
        "parse_instance_fn": pcenter_parse_instance,
        "feasibility_fn": pcenter_check_feasibility,
        "objective_model_fn": pcenter_objective_model,
    },
    "OSP": {
        "parse_instance_fn": osp_parse_instance,
        "feasibility_fn": osp_check_feasibility,
        "objective_model_fn": osp_objective_model,
    },
    "CSP": {
        "parse_instance_fn": csp_parse_instance,
        "feasibility_fn": csp_check_feasibility,
        "objective_model_fn": csp_objective_model,
    },
    "KMST": {
        "parse_instance_fn": kmst_parse_instance,
        "feasibility_fn": kmst_check_feasibility,
        "objective_model_fn": kmst_objective_model,
    },
    "JSP": {
        "parse_instance_fn": jssp_parse_instance,
        "feasibility_fn": jssp_check_feasibility,
        "objective_model_fn": jssp_objective_model,
    },
    "FSP": {
        "parse_instance_fn": fsp_parse_instance,
        "feasibility_fn": fsp_check_feasibility,
        "objective_model_fn": fsp_objective_model,
    },
    "CMP": {
        "parse_instance_fn": cmp_parse_instance,
        "feasibility_fn": cmp_check_feasibility,
        "objective_model_fn": cmp_objective_model,
    },
    "CFLP": {
        "parse_instance_fn": cflp_parse_instance,
        "feasibility_fn": cflp_check_feasibility,
        "objective_model_fn": cflp_objective_model,
    },
    "BPP": {
        "parse_instance_fn": bpp_parse_instance,
        "feasibility_fn": bpp_check_feasibility,
        "objective_model_fn": bpp_objective_model,
    },
    "AP3": {
        "parse_instance_fn": ap3_parse_instance,
        "feasibility_fn": ap3_check_feasibility,
        "objective_model_fn": ap3_objective_model,
    },
    "QSPP": {
        "parse_instance_fn": qspp_parse_instance,
        "feasibility_fn": qspp_check_feasibility,
        "objective_model_fn": qspp_objective_model,
    },
    "LOP": {
        "parse_instance_fn": lop_parse_instance,
        "feasibility_fn": lop_check_feasibility,
        "objective_model_fn": lop_objective_model,
    },
    "2SP": {
        "parse_instance_fn": twosp_parse_instance,
        "feasibility_fn": twosp_check_feasibility,
        "objective_model_fn": twosp_objective_model,
    },


}
