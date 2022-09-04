# To run the server, use the server.py script and streamlit
# set PROLIFIC_PID in the URL to care about Prolific IDs and redirection

streamlit server.py \
    --server.port 6666 \
    --config my_experiment/task_config.yaml \
    --instructions instructions.md \
    --name my_study_name 
    --inputs my_experiment/outputs/predictions1.json my_experiment/outputs/predictions2.json

# To process outputs of the study, run the processing script  
# performs Friedman test to check equality of mean rankings
# and Nemenyi post-hoc test to check pairwise differences

python process_results.py -n human-eval-in-house 