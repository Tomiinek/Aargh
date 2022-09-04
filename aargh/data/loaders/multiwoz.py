import os
import json
from aargh.data.loaders import Loader


class MultiWOZ(Loader):
    NAME = "multiwoz"
    VERSION = "2.2"
    DOMAINS = ["restaurant", "attraction", "hotel", "taxi", "train", "bus", "hospital", "police"]
    FILES = {"train" : 17, "dev" : 2, "test" : 2}
    RESOURCES = ["https://raw.githubusercontent.com/Tomiinek/multiwoz/master/data/MultiWOZ_2.2/dialog_acts.json",
                 "https://raw.githubusercontent.com/Tomiinek/multiwoz/master/data/MultiWOZ_2.2/goals.json"] + [
                { 
                    "file" : f"https://raw.githubusercontent.com/Tomiinek/multiwoz/master/data/MultiWOZ_2.2/{fold}/dialogues_{str(f_num + 1).zfill(3)}.json",
                    "target_dir" : fold 
                } for fold, num_files in FILES.items() for f_num in range(num_files)
                ] + [f"https://raw.githubusercontent.com/Tomiinek/multiwoz/master/db/{d}_db.json" for d in DOMAINS]

    @staticmethod
    def _get_file_items(file_path):
        with open(file_path) as f:
            return json.load(f)

    def _get_fold(self, fold_name, action_data=None):
        """
        Reads input .json file.
        
        Returns an object with the follwing format:

        [{  dialogue_id: Dialog ID, the name of original MultiWOZ 2.0 file is used,
            services: [ List of domain names that occured in this dialog. ]
            turns: [
                { turn_id: A 0-based index indicating the order of the utterances in the conversation,
                  speaker: Either USER or SYSTEM, indicating which role generated this utterance,
                  utterance: The raw text of the utterance,
                  frames: [{
                      actions : This is not used, but could possibly contain API calls. It is instead in the `dialog_acts.json` file?
                      service : Domain of the frame.
                      state: {
                          active_intent: String. User intent of the current turn. NONE if not present at all.
                          requested_slots: [ List of string representing the slots, the values of which are being requested by the user. ]
                          slot_values: { slot name : [ List of values ] }
                      },
                      slots : [{
                          slot : String of the slot name.
                          start: Starting character index in the utterance.
                          exclusive_end: The character character just after the end of the slot value in the utterance. Use utt[start:exclusive_end].
                          copy_from: The slot to copy from.
                          value: String of value. It equals to utterance[start:exclusive_end], where utterance is the current utterance in string.
                      }, ... ]
                  }, ... ],
                  dialog_act: {
                    action name : [ [ Argumnets ... ], ... ]
                  },
                  span_info: [ [ (Action name, value, start_index, end_index), ... ], ... ]
            }, ... ]
        }, ...]
        """

        items = []
        num_files = self.FILES[fold_name]

        for i in range(num_files):
            items.extend(MultiWOZ._get_file_items(os.path.join(self.get_root(), fold_name, f"dialogues_{str(i + 1).zfill(3)}.json")))

        if action_data is not None:

            item_mapping = { x["dialogue_id"].split(".")[0].lower() : (i, { y["turn_id"] : j for j, y in enumerate(x["turns"])}) for i, x in enumerate(items)}
            for action_key in action_data:
                short_action_key = action_key.split(".")[0].lower()

                if short_action_key not in item_mapping:
                    continue

                actions = action_data[action_key]
                for turn_id, v in actions.items():
                    
                    if turn_id not in item_mapping[short_action_key][1]:
                        continue

                    m = item_mapping[short_action_key]
                    turn = items[m[0]]["turns"][m[1][turn_id]]
                    action = actions[turn_id]

                    turn["dialog_act"] = action["dialog_act"]
                    turn["span_info"] = action["span_info"]
            
        return items

    def read(self, verbose=False):

        with open(os.path.join(self.get_root(), "dialog_acts.json"), "r") as f:
            action_data = json.load(f)
        
        train = self._get_fold('train', action_data)
        val = self._get_fold('dev', action_data)
        test = self._get_fold('test', action_data)

        return train, val, test

    def read_goals(self):

        booked_domains = {}
        goals = None

        with open(os.path.join(self.get_root(), "goals.json"), "r") as f:
            goals = json.load(f)

        for dialog in goals:
            booked_domains[dialog] = goals[dialog].pop('turn_booked_domains')

        return goals #, booked_domains