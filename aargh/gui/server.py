import string
import random
import json
import argparse
import time
import streamlit as st
from bokeh.models.widgets import Div
from aargh.data.abstract import AutoTask
from aargh.config import Params


def initialize():
    st.set_page_config(
        page_title="Dialog evaluation",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
        #menu_items={}
    )


@st.cache(suppress_st_warning=True)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to task config file.")
    parser.add_argument("-i", "--inputs", nargs='+', type=str, required=True, help="List of paths to model predictions (JSON).")
    parser.add_argument("-s", "--instructions", type=str, required=True, help="Path to the markdown file with a study description & instructions.")
    parser.add_argument("--max-convs", type=int, default=2, help="Maximal number of conversations shown to the participant.")
    parser.add_argument("--max-length", type=int, default=16, help="Soft upper bound for the number of dialog turns (in total) shown to the participant.")
    parser.add_argument("--sanity-num", type=int, default=3, help="Number of turns that will be used as sanity checks.")
    parser.add_argument("--sanity-sentence", type=str, nargs='+', default=[
        "Given this typing of entities and predicates, we arrive at a tentative definition of a generic sentence.",
        "The larger forms designated by special generic terms include the following.",
        "Suddenly he stopped at the foot of a tree."
    ], help="The sentece used for all sanity checks.")
    parser.add_argument("-n", "--name", type=str, required=True, help="Name of the evaluation, this is used as the output directory name.")
    parser.add_argument("-r", "--root", type=str, default="outputs", help="Root directory.")
    return parser.parse_args()


@st.cache(allow_output_mutation=True)
def load_data(task_config_path):
    params = Params.from_file(task_config_path)
    task_class = AutoTask.from_config(params, return_instance=False)
    task_class.prepare_data()
    task_class.setup()
    task = task_class(params, is_testing=True)
    task.transforms = None
    
    test_data = {}
    for i in task:
        if i.conv_id not in test_data:
            test_data[i.conv_id] = []
        test_data[i.conv_id].append(i)

    conversations = {}
    for file_path in args.inputs:
        with open(file_path, 'r') as f:
            conversations[file_path] = json.load(f)
    conversations_ids = [set(conversations[c].keys()) for c in conversations]
    conversations_ids.append(set(test_data.keys()))
    valid_dialog_ids = list(set.intersection(*conversations_ids))

    return valid_dialog_ids, conversations, test_data, args.inputs


def prepare_data(args):

    if 'eval_ids' in st.session_state and \
       'eval_conversations' in st.session_state and \
       'test_conversations' in st.session_state:  
       return

    with st.spinner('Lading data, please wait ...'):
        valid_dialog_ids, conversations, test_data, system_names = load_data(args.config)
        target_length = args.max_length
        current_length = 0
        eval_dialog_ids = []
        
        for _ in range(args.max_convs):
            
            max_length = target_length - current_length
            if max_length <= 3:
                break

            weights = []
            for id in valid_dialog_ids:
                real_length = len(test_data[id])
                w = max_length / max(1, abs(max_length - real_length))
                weights.append(w)

            for id in eval_dialog_ids:
                id_idx = valid_dialog_ids.index(id)
                weights[id_idx] = 0.0
            
            r = random.choices(valid_dialog_ids, weights, k=1)[0]
            current_length += len(test_data[r])
            eval_dialog_ids.append(r)

    st.session_state['eval_ids'] = eval_dialog_ids
    st.session_state['eval_conversations'] = conversations
    st.session_state['test_conversations'] = {i : test_data[i] for i in eval_dialog_ids}
    st.session_state['system_names'] = system_names


def run_application(args):

    def conversation_len(id):
        return len(st.session_state['test_conversations'][id])

    #
    # Side bar & submitting

    css = """

        body, div {
            font-family: sans-serif !important;
        }

        ul > li {
            font-size: 1.0rem !important;
            font-weight: 100;
        }

        .stRadio {
            margin-bottom: -20px
        }

        .stRadio > div > label:first-of-type {
            display: none
        }
        
        .stRadio > label {
            float: left
        }
        
        .stRadio > div {
            position: relative;
            float: left;
            flex-direction: row;
            justify-content: space-around;
            width: 280px;
        }

        .stRadio > div:after {
            content: '←  Worst';
            position: absolute;
            left: 100%;
            top: 1px;
            width: 75px;
            font-size: 0.9rem
        }

        .stRadio > div > label {
            background: rgb(40, 40, 40)
        }

        .stRadio > div > label:hover {
            background: rgb(60, 60, 60)
        }

        .stRadio > div > label > div:first-of-type {
            position: relative;
            left: 25px;
        }
        
        .stRadio > div > label > div:last-of-type {
            position: relative;
            left: -28px;
        }

        [data-testid="stSidebar"] .stButton {
            width: 100% !important;
        }

        .stButton > button {
            height: 50px;
            padding: 0 20px;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100% !important;
        }

        [data-testid="stSidebar"] > div:first-of-type {
            padding-top: 1.5rem;
        }

        .main > div {
            padding-top: 1.0rem;
        }

        .stProgress div {
            height: 0.3rem
        }

        .stAlert > [data-baseweb="notification"] {
            padding: 0.25rem 0.45rem;
            margin-bottom: -0.4rem;
        }

        #your-response-ranking {
            margin-top: 1.25rem;
        }
    
        #past-dialog-utterances {
            margin-top: 0.0rem;
        }

        [data-testid="stForm"] > div > [data-testid="stBlock"] {
            border: solid 1px rgb(70,70,70);
            border-radius: 5px;
            padding: 0.4rem 0.6rem 0.2rem 0.6rem;
        }

        .main [data-testid="stForm"] > div {
            width: 100% !important;
        }

        .main [data-testid="stForm"] > div > .element-container {
            width: 180px !important;
            display: inline-block;
            margin-bottom: 0;
        }

        .main [data-testid="stForm"] > div > .element-container div {
            width: 100% !important;
            display: inline-block
        }

        .main [data-testid="stImage"] {
            margin: 1rem auto;
            max-width: 900px;
        }

        .element-container [data-testid="stImage"] {
            margin: 0;
            left: -80px;
            position: relative;
            top: 35px; 
        }

        #MainMenu {visibility: hidden;}
        #footer {visibility: hidden;}
    """
    st.markdown('<style>' + css + '</style>', unsafe_allow_html=True)
     
    st.sidebar.title('Dialog evaluation')
    
    if 'introduced' not in st.session_state or not st.session_state['introduced']:

        def set_introduced():
            st.session_state['introduced'] = True

        st.sidebar.markdown("Welcome! :sparkles: Please read the study instructions before the start.")

        def show_markdown(md):
            md = md.replace("[N]", str(len(st.session_state['system_names'])))
            md = md.replace("[C]", str(len(st.session_state['eval_ids'])))
            # estimated time per turn is 20s
            # overhead for reading instructions is 2 minutes
            md = md.replace("[T]", str(2 + sum(20 * conversation_len(dialog_id) for dialog_id in st.session_state['eval_ids']) // 60))
            
            st.markdown(md)

        with open(args.instructions, 'r') as f:
            line_buffer = []
            for line in f:
                if "[IMG]" in line:
                    show_markdown(''.join(line_buffer))
                    line_buffer.clear()
                    st.image(line.replace('[IMG]', "").strip())
                    continue
                line_buffer.append(line)
            show_markdown(''.join(line_buffer))
        
        st.button("I understand the task, let's start!", on_click=set_introduced)
        st.stop()
    
    button_placeholders = [st.sidebar.empty() for i in range(len(st.session_state['eval_ids']))]
    
    with st.sidebar.form(key='my-form'):
        note = st.text_area("Leave us a note:")
        warning_placeholder = st.empty()
        final_submit = st.form_submit_button('Submit the survey')
    
    submitted = False
    if final_submit:

        submitted = True

        completed = True
        for dialog_id in st.session_state['eval_ids']:
            c = sum(x != 0 for v in st.session_state['ratings'][dialog_id].values() for x in v) // len(st.session_state['system_names'])
            t = conversation_len(dialog_id)
            completed = completed and (c == t)
            if not completed:
                break

        if completed:
            
            st.title("Thank you!")
            st.balloons()

            from_prolific = True
            params = st.experimental_get_query_params()
            participant_id = params.get("PROLIFIC_PID", [None])[0]
            if participant_id is None:
                from_prolific = False
                participant_id = params.get("user", [""])[0]
                participant_id = participant_id + "_" + ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(6))

            output_directory = os.path.join(args.root, args.name)
            output_filename = participant_id + '.json'

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            with open(os.path.join(output_directory, output_filename), 'w+') as f:
                json.dump({
                    "note" : note,
                    "system_names" : st.session_state['system_names'],
                    "dialog_ids" : st.session_state['eval_ids'],
                    "ratings" : st.session_state['ratings'],
                    "sanity_checks" : st.session_state['sanity_checks']
                }, f, indent=2)

            if from_prolific:
                wait_time = 3
                status_text = st.empty()
                for i in range(wait_time):
                    status_text.success(f"You will be redirected to Prolific in {wait_time - i} s!")
                    time.sleep(1)

                js = "window.location.href = 'https://app.prolific.co/submissions/complete?cc=7E1D8487'" 
                js = "location.reload();"
                html = '<img src onerror="{}">'.format(js)
                div = Div(text=html)
                st.bokeh_chart(div)
                st.stop()
            else:
                wait_time = 3
                status_text = st.empty()
                for i in range(wait_time):
                    status_text.success(f"A new batch of conversations will be shown in {wait_time - i} s!")
                    time.sleep(1)

                js = "location.reload();"
                html = '<img src onerror="{}">'.format(js)
                div = Div(text=html)
                st.bokeh_chart(div)
                st.stop()

            return

    if submitted:
        warning_placeholder.error("You have to fill in the study first!")

    # 
    # Conversation page

    if 'turn_idx' not in st.session_state:
        st.session_state['turn_idx'] = 0

    if 'conversation_selected' not in st.session_state:
        st.session_state['conversation_selected'] = [False] * len(st.session_state['eval_ids'])

    if 'ratings' not in st.session_state:
        st.session_state['ratings'] = {
            dialog_id : {n : [0] * conversation_len(dialog_id) for n in st.session_state['system_names']} for dialog_id in st.session_state['eval_ids'] 
        }

    if 'sanity_checks' not in st.session_state:
        st.session_state['sanity_checks'] = {
            dialog_id : {n : [False] * conversation_len(dialog_id) for n in st.session_state['system_names']} for dialog_id in st.session_state['eval_ids'] 
        }
        total_turns = sum(conversation_len(dialog_id) for dialog_id in st.session_state['eval_ids'])
        for r in random.sample(range(total_turns), int(args.sanity_num)):
            offset = 0
            for dialog_id in st.session_state['eval_ids']:
                if r >= offset + conversation_len(dialog_id):
                    offset += conversation_len(dialog_id)
                    continue
                n = random.choice(st.session_state['system_names'])
                st.session_state['sanity_checks'][dialog_id][n][r - offset] = True
                break

    if 'turn_shuffler' not in st.session_state:
        
        def get_permutation():
            p = list(range(len(st.session_state['system_names'])))
            random.shuffle(p)
            return p

        st.session_state['turn_shuffler'] = {
            dialog_id : [get_permutation() for _ in range(conversation_len(dialog_id))] for dialog_id in st.session_state['eval_ids'] 
        }

    def select_conversation():
        st.session_state['conversation_selected'] = [getattr(st.session_state, f"selection_{id}") for id in st.session_state['eval_ids']]
        st.session_state['turn_idx'] = 0
        st.session_state['reached_end'] = False

    for i, dialog_id in enumerate(st.session_state['eval_ids']):
        dialog_id = st.session_state['eval_ids'][i]
        completed = sum(x != 0 for v in st.session_state['ratings'][dialog_id].values() for x in v) // len(st.session_state['system_names'])
        total = conversation_len(dialog_id)
        button_placeholders[i].button(f'Conversation {i+1} ({completed}/{total} done)', key=f"selection_{dialog_id}", on_click=select_conversation)

    # 
    # Turn page

    for idx, c in enumerate(st.session_state['conversation_selected']):
        if not c:
            continue

        conversation_id = st.session_state['eval_ids'][idx]
        turn_idx = st.session_state['turn_idx']
        num_systems = len(st.session_state['system_names'])
        

        st.write("#### 🗨️ &nbsp; Dialogue context:")
        st.progress(turn_idx / (conversation_len(conversation_id) - 1))

        turn_data = st.session_state['test_conversations'][conversation_id][turn_idx]
        for i, past_turn in enumerate(turn_data.context):
            if past_turn['speaker'] == "user":
                st.info(f"*{i + 1}. USR:* {past_turn['utterance']}")
            else:
                st.warning(f"*{i + 1}. SYS:* {past_turn['utterance']}")
        db_results = [f"{domain}: {len(turn_data.api_result[domain]['results'])}" for domain in turn_data.api_result]
        db_results = ', '.join(db_results) if len(db_results) > 0 else "none"
        st.error(f"*Available database entries:* **{db_results}**")

        st.write("#### 🏆 &nbsp; Your response ranking:")    
        #st.write("Plase, sort the following sentences from the best-fitting (with respect to the context) to the worst.")
        #st.write("You can rank multiple sentencec the same, but please be as decisive as you can.")

        def prev_turn():
            if st.session_state['turn_idx'] > 0:
                st.session_state['turn_idx'] -= 1
            st.session_state['reached_end'] = False

        def next_turn():
            if st.session_state['turn_idx'] < conversation_len(conversation_id) - 1:
                st.session_state['turn_idx'] += 1      
            else:
                st.session_state['reached_end'] = True
            for n in st.session_state['system_names']:
                rating = getattr(st.session_state, f"rating_{conversation_id}_{turn_idx}_{n}")
                if rating != "0.":
                    st.session_state['ratings'][conversation_id][n][turn_idx] = int(rating[:-1])
                
        if st.session_state['reached_end']:
            st.success("**End of conversation, please select other incompleted one in the sidebar!**")
        
        with st.form(key='my_form'):

            #cols = st.columns(num_systems)
            permutation = st.session_state['turn_shuffler'][conversation_id][turn_idx]
            
            for i in range(len(st.session_state['system_names'])):
                #with cols[permutation[i]]:
                i = permutation[i]
                n = st.session_state['system_names'][i]
                c = st.container()
                t_data = st.session_state['eval_conversations'][n][conversation_id][turn_idx]
                response = random.choice(args.sanity_sentence) if st.session_state['sanity_checks'][conversation_id][n][turn_idx] else t_data['response_raw'][0]
                selected_idx = st.session_state['ratings'][conversation_id][n][turn_idx]
                c.radio("Best →", [str(x) + '.' for x  in range(num_systems + 1)], index=selected_idx, key=f"rating_{conversation_id}_{turn_idx}_{n}")
                c.markdown(f"**{response}**")
                # c.write(n)

            #left, right, _ = st.columns([8, 8, 53])
            st.form_submit_button("« previous turn", on_click=prev_turn)
            st.form_submit_button("Submit" if turn_idx == conversation_len(conversation_id) - 1 else "Submit & next »", on_click=next_turn)

        break
    else:
        st.success("No conversation selected, please select it in the sidebar")
        # with st.container():
        #    st.image("guis/arrow.gif")



if __name__ == '__main__':

    initialize()
    args = parse_args()
    prepare_data(args)
    run_application(args)
