"""Extra Args for NLP / GLUE experiments (parsed together with vision Args)."""

from utility.args import Args


def register_nlp_glue_args() -> None:
    Args.add_argument("--glue_task", type=str, help="GLUE config name: cola,sst2,mrpc,qqp,mnli,qnli,rte,wnli,stsb")
    Args.add_argument("--glue_output_root", type=str, help="Root directory for GLUE logs and metrics JSON")
