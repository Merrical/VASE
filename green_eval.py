import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
from green_score import GREEN


def eval_t2t_csv(in_csv_file, out_csv_file):
    model_name = "StanfordAIMI/GREEN-radllama2-7b"
    green_scorer = GREEN(model_name, output_dir=".")
    in_df = pd.read_csv(in_csv_file, usecols=[0, 1, 2, 3])

    in_df['ref_answer'] = in_df['question'] + " " + in_df['ref_answer']
    in_df['gen_answer'] = in_df['question'] + " " + in_df['gen_answer']
    refs_list = in_df['ref_answer'].tolist()
    hyps_list = in_df['gen_answer'].tolist()

    mean, std, green_score_list, summary, result_df = green_scorer(refs_list, hyps_list)

    result_df = result_df.drop(['reference', 'predictions'], axis=1)
    in_df.reset_index(drop=True, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    out_df = pd.concat([in_df, result_df], axis=1)

    out_df.to_csv(out_csv_file, index=False)

# conda activate env_green (!!! Set up a dedicated virtual environment for the Green model to prevent conflicts with environments used for other MLLMs.)
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 green_eval.py
# deactivate
if __name__ == "__main__":
    in_csv_file = "outputs/radvqa_medgemma_hallscore.csv"
    out_csv_file = "outputs/radvqa_medgemma_green.csv"
    eval_t2t_csv(in_csv_file, out_csv_file)