import re
import hashlib
from collections import deque, Counter, OrderedDict
from fuzzywuzzy import fuzz
from torchvision import transforms
from titlecase import titlecase
from aargh.utils.data import time_str_to_minutes
from aargh.utils.logging import get_logger, highlight
from aargh.data.tasks.base import BaseGoalTask
from aargh.data.tasks.goal_chat import LanguageModelGoalTask, DoubleLanguageModelGoalTask, PolicyOptimizationLanguageModelGoalTask
from aargh.data.transforms import LinearizeAttributePairTransform, LinearizeAttributeTransform, PrefixAttributeTransform
from aargh.data.databases import MultiWOZDatabase


class MultiWOZTask(BaseGoalTask):

    VAL_SIZE = 0
    DATASETS = [("multiwoz", "2.2")]

    def __init__(self, params, *args, **kwargs):
        self.delexicalized_span_re = re.compile(r'\[(\s*[\w_\s]+)\s*\]')
        self.database = MultiWOZDatabase()
        super().__init__(params, *args, **kwargs)

    def get_cache_filename(self):
        to_hash = str(self.params.context_length)
        params_hash = hashlib.md5(to_hash.encode()).hexdigest()         
        return self.NAME + "_" + params_hash + ("_test" if self.is_testing else "_devel")

    def _data_to_items(self, data, is_testing):

        train_data, val_data, test_data = data[self.DATASETS[0]]
        
        if not is_testing:
            train = self._fold_to_items(train_data)
            val = self._fold_to_items(val_data) 
            test = None
        else:
            train, val = None, None
            test = self._fold_to_items(test_data)

        return train, val, test

    def _fold_to_items(self, data):

        def flatten(list_of_lists):
            out_list = []
            size = 0
            for sublist, s in list_of_lists:
                size += s
                out_list.extend(sublist)
            return out_list, size

        def process_chunk(chunk_data):
            chunk_items = []
            chunk_rejected = 0
            for conv in chunk_data:
                accepted_items, num_rejected_items = self._conversation_to_items(conv) 
                chunk_items.extend(accepted_items)
                chunk_rejected += num_rejected_items
            return chunk_items, chunk_rejected

        #n_jobs = get_num_cpus()
        #items, total_rejected_items = preprocess_parallel(data, process_chunk, n_jobs=n_jobs, chunksize=len(data)//n_jobs+1, flatten_func=flatten)

        total_rejected_items = 0
        items = []
        for conv in data:
            accepted_items, num_rejected_items = self._conversation_to_items(conv) 
            items.extend(accepted_items)
            total_rejected_items += num_rejected_items

        for idx, i in enumerate(items):
            i.idx = idx

        if total_rejected_items > 0:
            get_logger(self.NAME).warning(f"Rejected {highlight(total_rejected_items, c='r')} of {highlight(total_rejected_items + len(items))} dataset items!")
        
        return items

    def _lazy_prepare_item_hook(self, item):
        if item.api_result is not None:
            return 
        item.api_result = {}
        if item.api_call is not None:
            item.api_result = self._intents_to_db_results(item.api_call, item.belief, item.actions)
        item.api_call = {} # we alredy have results of the calls, they are not needed

    def _intents_to_db_results(self, domain, constraints, actions=None):
        result = self.api_call(domain, **constraints.get(domain, {}))       
        # change booking-successful to booking-unsuccessful if we want to book something and all required fields were provided
        # we need to do this, because MultiWOZ database does not contain information about booking availability
        # if result['booking'] and actions is not None and 'booking-nobook' in actions:
        #     result['booking'] = False
        if actions is not None and 'booking-nobook' in actions:
            result['results'] = []
        return { domain: result }

    #@staticmethod
    #def get_new_tokens():
    #    all_delex_spans = ['bookstay', 'duration', 'name', 'trainid', 'stars', 'destination', 'address', 'entrancefee', 
    #                       'phone', 'pricerange', 'price', 'type', 'food', 'bookpeople', 'ref', 'choice', 'day', 'postcode', 
    #                       'arriveby', 'departure', 'leaveat', 'openhours', 'booktime', 'bookday', 'department', 'area'] 
    #    return [(f'[{x}]', False) for x in all_delex_spans]

    def api_call(self, api_name, **kwargs):
        """
        Perform an API call. Returns a pair of (list of results, booking information).
        The booking information is None if some fields required for booking are missing, True otherwise because the database
        does not provide any information about booking availability. The True can be cahnged to False based on the ground-truth
        system anotation of the particular turn, see `_intents_to_db_results`.

        Arguments:
            api_name (str): Name of the API.
            **kwargs: API call arguments (possibly hard constraints for the database search).
        """

        # Listing of possible API calls with arguments:

        # find_hotel - pricerange, type, parking, bookday, bookpeople, bookstay, stars, internet, name, area
        # find_train - destination, arriveby, departure, day, bookpeople, leaveat
        # find_bus - destination, departure, day, leaveat
        # find_attraction - area, name, type
        # find_restaurant - pricerange, area, food, name, bookday, bookpeople, booktime
        # find_hospital - department
        # police - name

        # book_hotel - pricerange, type, parking, bookday, bookpeople, bookstay, stars, internet, name, area
        # book_train - destination, arriveby, departure, day, bookpeople, leaveat
        # book_restaurant - pricerange, area, food, name, bookday, bookpeople, booktime
        # find_taxi - leaveat, destination, departure, arriveby

        domain = api_name
        
        # hacking naming that is in the dataset and does not pass the fuzzy database search
        # and removing dontcare keys
        keys_to_remove = []
        for k, v in kwargs.items():
            if 'dontcare' in v:
                keys_to_remove.append(k)
                continue
            if k == 'name' or (domain == 'taxi' and (k == 'departure' or k == 'destination')):
                kwargs[k] = [self._name_to_canonical(x) for x in v]  
            elif k == 'food':
                kwargs[k] = [self._food_to_canonical(x) for x in v]
        for k in keys_to_remove:
            kwargs.pop(k)

        # ignore other search constraints if we are **training** and the name is present (because of broken annotation)
        if not self.is_testing:
            if domain in ['hotel', 'restaurant', 'attraction'] and 'name' in kwargs:
                kwargs = {
                    k : v for k, v in kwargs.items() if k == 'name' or k not in self.database.data_keys[domain]
                }

        if domain == 'hotel':
            if 'parking' in kwargs and kwargs['parking'] == "free":
                kwargs['parking'] = "yes"
            if 'internet' in kwargs and kwargs['internet'] == "free":
                kwargs['internet'] = "yes"

        results = self.database.query(domain, constraints=kwargs)

        # sort search results if we care about arrivals and departure times ...
        if "leaveat" in kwargs and domain == 'train':
            results.sort(key=(lambda x: time_str_to_minutes(x["leaveat"])))
        elif "arriveby" in kwargs and domain == 'train':
            results.sort(key=(lambda x: time_str_to_minutes(x["arriveby"])), reverse=True)

        # booking_available = None          
        # if domain == "hotel":
        #     if "bookday" in kwargs and "bookpeople" in kwargs and "bookstay" in kwargs:
        #         booking_available = True
        # elif domain == "restaurant":
        #     if "bookday" in kwargs and "bookpeople" in kwargs and "booktime" in kwargs:
        #         booking_available = True
        # elif domain == "train":
        #     if "destination" in kwargs and "departure" in kwargs and "bookpeople" in kwargs and \
        #        "day" in kwargs and ("leaveat" in kwargs or "arriveby" in kwargs):
        #         booking_available = True
        # elif domain == "taxi":
        #     if "destination" in kwargs and "departure" in kwargs and ("leaveat" in kwargs or "arriveby" in kwargs):
        #         booking_available = True
        
        # if booking_available and len(results) == 0:
        #     booking_available = False

        # multiply price database results with the number of people
        if "bookpeople" in kwargs:
            price_re = re.compile(r'\d+(\.\d+)?')
            
            price_name = ""
            if domain == "train": 
                price_name = "price"
            elif domain == "attraction": 
                price_name = "entrancefee" 
            
            for r in results:
                if (domain == "train" and "price" not in r) or \
                   (domain == "attraction" and "entrancefee" not in r) or \
                   (domain not in ["train", "attraction"]):
                    continue
                if kwargs["bookpeople"][0].isdigit():
                    num_people = int(kwargs["bookpeople"][0])
                else:
                    num_people = 1
                
                def multiply_people(m):
                    price = float(m.group(0)) * num_people
                    return format(price, '.2f')

                r[price_name] = price_re.sub(multiply_people, r[price_name])
            
        # remove keys from results which were used for querying, because they are not interesting
        # for r in results:
        #     for k in kwargs:
        #         if k in ["leaveat", "arriveby"]:
        #             continue
        #         if k in r:
        #             r.pop(k)
        #     if 'ref' in r and not is_booking:
        #         r.pop('ref')

        return {
            "results" : results,
            "booking" : None
        } 

    def _name_to_canonical(self, name, domain=None):
        """ Converts name to another form which is closer to the canonical form used in database. """

        name = name.strip().lower()
        name = name.replace(" & ", " and ")
        name = name.replace("&", " and ")
        
        if domain is None or domain == "restaurant":
            if name == "hotel du vin bistro":
                return "hotel du vin and bistro"
            elif name == "the river bar and grill":
                return "the river bar steakhouse and grill"
            elif name == "nando's":
                return "nandos"
            elif name == "city center b and b":
                return "city center north b and b"
            elif name == "acorn house":
                return "acorn guest house"
            elif name == "caffee uno":
                return "caffe uno"
            elif name == "cafe uno":
                return "caffe uno"
            elif name == "rosa's":
                return "rosas bed and breakfast"
            elif name == "restaurant called two two":
                return "restaurant two two"
        
        if domain is None or domain == "hotel":
            if name == "lime house":
                return "limehouse"
            elif name == "cityrooms":
                return "cityroomz"
            elif name == "whale of time":
                return "whale of a time"
            elif name == "huntingdon hotel":
                return "huntingdon marriott hotel"
            elif name == "holiday inn exlpress, cambridge":
                return "express by holiday inn cambridge"
            elif name == "university hotel":
                return "university arms hotel"
            elif name == "arbury guesthouse and lodge":
                return "arbury lodge guesthouse"
            elif name == "bridge house":
                return "bridge guest house"
            elif name == "arbury guesthouse":
                return "arbury lodge guesthouse"
            elif name == "nandos in the city centre":
                return "nandos city centre"
        
        if domain is None or domain == "attraction":
            if name == "broughton gallery":
                return "broughton house gallery"
            elif name == "scudamores punt co":
                return "scudamores punting co"
            elif name == "cambridge botanic gardens":
                return "cambridge university botanic gardens"
            elif name == "the junction":
                return "junction theatre"
            elif name == "trinity street college":
                return "trinity college"
            elif name in ['christ college', 'christs']:
                return "christ's college"
            elif name == "history of science museum":
                return "whipple museum of the history of science"
            elif name == "parkside pools":
                return "parkside swimming pool"
            elif name == "the botanical gardens at cambridge university":
                return "cambridge university botanic gardens"
            elif name == "cafe jello museum":
                return "cafe jello gallery"

        return name

    def _time_to_canonical(self, time):
        """ Converts time to the only format supported by database, e.g. 07:15. """

        time = time.lower()
                    
        if time == "afternoon": return "13:00"
        if time == "lunch" or time == "noon" or time == "mid-day" or time == "around lunch time": return "12:00"
        if time == "morning": return "08:00"
        if time.startswith("one o'clock p.m"): return "13:00"
        if time.startswith("ten o'clock a.m"): return "10:00"
        if time == "seven o'clock tomorrow evening":  return "07:00"
        if time == "three forty five p.m":  return "15:45"
        if time == "one thirty p.m.":  return "13:30"
        if time == "six fourty five":  return "06:45"
        if time == "eight thirty":  return "08:30"

        if time.startswith("by"):
            time = time[3:]

        if time.startswith("after"):
            time = time[5:].strip()

        if time.startswith("afer"):
            time = time[4:].strip()    

        if time.endswith("am"):   time = time[:-2].strip()
        if time.endswith("a.m."): time = time[:-4].strip()

        if time.endswith("pm") or time.endswith("p.m."):
            if time.endswith("pm"):   time = time[:-2].strip()
            if time.endswith("p.m."): time = time[:-4].strip()
            tokens = time.split(':')
            if len(tokens) == 2:
                return str(int(tokens[0]) + 12) + ':' + tokens[1] 
            if len(tokens) == 1 and tokens[0].isdigit():
                return str(int(tokens[0]) + 12) + ':00'

        if time[-1] == '.' or time[-1] == ',' or time[-1] == '?':
            time = time[:-1]

        if time.isdigit(): return time.zfill(2) + ":00"

        if len(time) == 4 and time[1] == ':':
            tokens = time.split(':')
            return tokens[0].zfill(2) + ':' + tokens[1]

        return time

    def _food_to_canonical(self, food):
        """ Converts food name to caninical form used in database. """
        
        food = food.strip().lower()

        if food == "eriterean": return "mediterranean"
        if food == "brazilian": return "portuguese"
        if food == "seafood": return "sea food"
        if food == "portugese": return "portuguese"
        if food == "modern american": return "north american"
        if food == "americas": return "north american"
        if food == "intalian": return "italian"
        if food == "italain": return "italian"
        if food == "asian or oriental": return "asian"
        if food == "english": return "british"
        if food == "australasian": return "australian"
        if food == "gastropod": return "gastropub"
        if food == "brutish": return "british"
        if food == "bristish": return "british"

        return food
  
    def _match_slot_value_to_conversation(self, options, slots, utterance, context, last_belief):
        """ Search for a support of the annotated slot-value pair in the conversation context. """
        
        # search in the last dialog state
        if self.differential_belief:
            for d in last_belief:
                for s in last_belief[d]:
                    for o in options:
                        if o.strip().lower() in last_belief[d][s]:
                            return o

        # search in span anotations (slots) of the frame
        for s in slots:
            for slot_value in s["value"]:
                slot_value = slot_value.strip().lower()
                for o in options:
                    if slot_value == o.strip().lower():
                        return o

        # search in the current utterance
        for o in options:
            if utterance.lower().find(o.strip().lower()) != -1:
                return o

        # search in the context
        for u in reversed(context):
            for o in options:
                if u["utterance"].lower().find(o.strip().lower()) != -1:
                    return o
        
        return None

    def _delexicalize_utterance(self, utterance, span_info, is_testing):

        span_info.sort(key=(lambda  x: x[-2])) # sort spans by start index
        new_utterance = ""
        prev_start = 0

        for span in span_info:
            intent, slot_name, value, start, end = span
            if start < prev_start or value == "dontcare":
                continue
            if not is_testing and utterance[start:end].lower() != value.lower():
                return None
            new_utterance += utterance[prev_start:start]
            new_utterance += f"[{slot_name}]"
            prev_start = end

        new_utterance += utterance[prev_start:]
        return new_utterance 

    def _parse_slots(self, frame, utterance, context, last_belief):

        domain = frame["service"]
        state = frame["state"]
        slots = state["slot_values"]

        if len(slots) == 0:
            return domain, None

        new_slots = {}
        for name in slots:

            if "dontcare" in slots[name]:
                continue 

            if name.endswith("leaveat") or name.endswith("arriveby") or name.endswith("booktime"):
                times = [self._time_to_canonical(x) for x in slots[name]] 
                proper_times = list(set([x for x in times if len(x) == 5 and x[2] == ':']))

                if len(proper_times) == 2:
                    time_1 = time_str_to_minutes(proper_times[0])
                    time_2 = time_str_to_minutes(proper_times[1])
                    if abs(time_1 - time_2) <= 15:
                        if time_2 > time_1: proper_times = [proper_times[0]]
                        if time_1 > time_2: proper_times = [proper_times[1]]
                    elif abs(time_1 - time_2) == 60*12:
                        if time_2 > time_1: proper_times = [proper_times[1]]
                        if time_1 > time_2: proper_times = [proper_times[0]]

                #elif len(proper_times) == 0 or len(proper_times) >= 2:
                #    return None, None
                
                slots[name] = proper_times

            elif name.endswith("name") or name == "taxi-destination" or name == "taxi-departure": 

                options = slots[name]

                # if there are more options, find similar pairs and merge them, there are no similar triples
                if len(options) > 1:
                    slots[name], sim_pairs = [], []
                    used_names = set()
                    
                    # find all similarities and mark names with single occurance
                    for i, n1 in enumerate(options[:-1]):
                        for j, n2 in enumerate(options[i+1:]):        
                            if fuzz.partial_ratio(self._name_to_canonical(n1), self._name_to_canonical(n2)) > 75:
                                used_names.add(i)
                                used_names.add(i + j + 1)
                                sim_pairs.append((n1, n2))

                    # add items with a single occurance to the resulting slot value
                    for i in range(len(options)):
                        if i not in used_names:
                            slots[name].append(options[i])
                            
                    # merge the pairs and select the form that is present in the conversation
                    for n1, n2 in sim_pairs:
                        valid_slots = ['hotel-name', 'restaurant-name', 'attraction-name', 'taxi-departure', 'taxi-destination']
                        match = self._match_slot_value_to_conversation([n1, n2], frame["slots"], utterance, context, last_belief)
                        if match is not None:
                            slots[name].append(match)
                        else:
                            return None, None
                
                elif len(options) == 1:
                    match = self._match_slot_value_to_conversation(options, frame["slots"], utterance, context, last_belief)
                    if match is None:
                        return None, None

            elif name.endswith("food"):

                options = slots[name]
                
                if len(options) == 2 and fuzz.partial_ratio(self._food_to_canonical(options[0]), self._food_to_canonical(options[1])) > 75:
                    match = self._match_slot_value_to_conversation(options, frame["slots"], utterance, context, last_belief)
                    if match is not None:
                        slots[name] = [match]
                    else:
                        return None, None
            else:
                if domain == "hospital" and len(slots[name]) > 1:
                    return None, None
       
            new_slots[name.split('-')[-1]] = slots[name]

        return domain, None if not new_slots else new_slots


    def _get_belief_difference(self, belief, old_belief):

        difference = {}
        domains = set(belief) | set(old_belief)

        for d in domains:
            difference[d] = {}

            slots = belief.get(d, {})
            old_slots = old_belief.get(d, {})

            new_slots = set(slots) - set(old_slots)

            for s in new_slots:
                difference[d][s] = slots[s]

            for s in set(old_slots) - set(slots):
                difference[d][s] = [self.remove_label]

            for s in set(slots) - new_slots:
                if slots[s] != old_slots[s]:
                    difference[d][s] = slots[s]

            if len(difference[d]) == 0 and d in old_belief:
                difference.pop(d)

        return difference

    def _conversation_to_items(self, conversation):

        def normalize_action(action):     
            normalized = {}
            for k, v in action.items():
                normalized[k.lower()] = [tuple(x) for x in v]      
            return normalized

        accepted_items, num_rejected_items = [], 0
        context = deque(maxlen=self.params.context_length)  

        belief, old_belief, old_results, prev_actions = {}, {}, {}, {}
        active_domain = None
        reject = False
        
        for turn in conversation["turns"]:
            if turn["speaker"].lower() == "system":
                
                turn_action = normalize_action(turn["dialog_act"])
                utterance = turn["utterance"]  
                belief_difference = self._get_belief_difference(belief, old_belief)     

                #
                # find active domain

                belief_active_domains = set(belief_difference.keys())
                action_active_domains = set()
                general_domain = False
                for action in turn_action:
                    domain, _ = action.split('-')
                    if domain in ['restaurant', 'hotel', 'taxi', 'train', 'hospital', 'police', 'attraction']:
                        action_active_domains.add(domain)
                    if domain == 'general':
                        general_domain = True

                possibly_active_domains = action_active_domains | belief_active_domains

                if len(possibly_active_domains) == 1:
                    active_domain = next(iter(possibly_active_domains))
                elif len(possibly_active_domains) > 1:
                    if active_domain in possibly_active_domains:
                        possibly_active_domains.remove(active_domain)
                        active_domain = next(iter(possibly_active_domains))
                    elif len(belief_active_domains) == 1:
                        active_domain = next(iter(belief_active_domains))
                    elif len(action_active_domains) == 1:
                        active_domain = next(iter(action_active_domains))
                    else:
                        active_domain = next(iter(possibly_active_domains))
                else:
                    if general_domain:
                        active_domain = None

                # 
                # delexicalize utterance and accept it only if it is possible to lexicalize it back

                if self.delexicalize and not reject:
                    delexicalized_utterance = self._delexicalize_utterance(utterance, turn["span_info"], self.is_testing)
                    if delexicalized_utterance is None:  
                        num_rejected_items += 1
                        continue
                    
                    if active_domain is not None:
                        results = self._intents_to_db_results(active_domain, belief, turn_action)
                    else:
                        results = {}
                    lexicalized = self._lexicalize(delexicalized_utterance, belief, results)
                    
                    if lexicalized is None and not self.is_testing:
                        lexicalized = self._lexicalize(delexicalized_utterance, belief, old_results)
                        if lexicalized is None:
                            reject = True
                    # else:
                    #     if lexicalized.lower() != turn["utterance"].lower():
                    #         get_logger(self.NAME).warning(f'{highlight(lexicalized)}')
                    #         get_logger(self.NAME).warning(f'{highlight(turn["utterance"])}')
                    old_results = results.copy()
        
                #
                # add new item if not rejecting

                if reject:
                    num_rejected_items += 1
                    reject = False
                    continue
                    
                target_belief = belief_difference if self.differential_belief else belief
                target_utterance = delexicalized_utterance if self.delexicalize else utterance

                target_belief = OrderedDict(sorted(target_belief.items()))

                # add the active domain as a key to the target belief
                if active_domain is not None:
                    if active_domain not in target_belief:
                        target_belief[active_domain] = {}
                    target_belief.move_to_end(active_domain, last=False)

                item = self.DatasetItem(None, conversation["dialogue_id"], target_utterance, list(context), old_belief, belief, target_belief, 
                                        active_domain, None, actions=turn_action, prev_actions=prev_actions)

                if self.delexicalize: # save also the original utterance as it can be useful during evaluation
                    item.response_raw = utterance

                accepted_items.append(item)  

            else:
                old_belief = belief.copy()
                belief = {}
                prev_actions = normalize_action(turn['dialog_act'])
                reject = False

                #
                # get the belief state

                for frame in turn["frames"]:                    
                    domain, slots = self._parse_slots(frame, turn["utterance"], context, old_belief)
                    if domain is None and slots is None and not self.is_testing:
                        reject = True
                    elif slots is None:
                        continue
                    else:           
                        if domain == "bus":
                            domain = "train"          
                        belief[domain] = slots
         
            context.append({'speaker': turn["speaker"].lower(), 'utterance': turn["utterance"]}) 

        return accepted_items, num_rejected_items

    def _lexicalize(self, utterance, belief, database_results):

        # find all delexicalized spans, i.e. [...]
        matches = list(self.delexicalized_span_re.finditer(utterance))

        if len(matches) == 0:
            return utterance

        span_names = set([m.group(1) for m in matches])
        all_domains = set(belief.keys()) | set(database_results.keys())

        # find the domain of most spans in the utterance (the on with greatest intersection in span names)
        max_intersection, max_count = 0, 0
        main_domain = None  
        for d in all_domains:
            
            belief_keys = belief.get(d, {}).keys()
            db_domain_results = database_results.get(d, {'results' : []})['results']
            db_keys = set() if len(db_domain_results) == 0 else db_domain_results[0].keys()
            intersection_size = len(set.intersection(span_names, set(belief_keys) | set(db_keys)))
    
            if intersection_size > max_intersection or (intersection_size == max_intersection and len(db_domain_results) > max_count):
                max_intersection = intersection_size
                max_count = len(db_domain_results)
                main_domain = d

        # no domain with known names seems to be suitable for the lexicalization
        if main_domain is None:
            return None

        def is_in_domain(name, domain):
            if name in belief.get(domain, {}):
                return True
            db = database_results.get(domain, {'results' : []})['results']
            if len(db) == 0:
                return False
            return name in db[0]

        def add_mapping(value):
            span_mappings.append((s, e, value))

        span_mappings = []      # to store the mappings from [start:end] to the lexicalized text
        last_num_choices = 1    # value of last choice span, used for adding s/es to nouns
        value_offsets = {n : -1 for n in span_names} # keeps track of already used items
        
        important_span_names = [m.group(1) for m in matches if m.group(1) in ['name', 'type', 'trainid', 'food']]
        if len(important_span_names) > 0:
            c = Counter(important_span_names)
            unique_count = c.most_common(1)[0][1]
            unique_span_name = max(c, key=c.get) # keeps span name that must be unique through the utterance
        else:
            unique_count = 1
            unique_span_name = ''
            for k in span_names:
                value_offsets[k] += 1

        used_values = set()     # values already used for filling unique_span_name

        for match in matches:
            s, e = match.start(), match.end()
            span_name = match.group(1)
            value_offsets[span_name] = min(value_offsets[span_name] + 1, unique_count - 1)
            offset = value_offsets[span_name]
            
            if span_name == "choice":
                num_results = len(database_results.get(main_domain, {'results' : []})['results'])
                last_num_choices = num_results

                if num_results >= 12 and utterance[e:].startswith(" of"):
                    add_mapping("dozens")
                elif num_results > 5:
                    add_mapping("several")
                else:
                    add_mapping(num_results)
                continue

            # if the span is not in current main domain, try to find it in a different one
            domain = main_domain
            if not is_in_domain(span_name, domain):
                for d in all_domains:
                    if is_in_domain(span_name, d):
                        domain = d
                        break

            # lexicalize span names related to booking, these are present only in the dialog state
            if span_name.startswith("book") and domain in belief:
                if span_name not in belief[domain] or len(belief[domain][span_name]) == 0:
                    return None
                offset = min(offset, len(belief[domain][span_name]) - 1)     
                if span_name == "bookday":
                    add_mapping(titlecase(belief[domain][span_name][offset])) 
                else:        
                    add_mapping(belief[domain][span_name][offset])

            # try to fill the span with a value from the database results
            elif domain in database_results and len(database_results[domain]['results']) > offset and span_name in database_results[domain]['results'][offset]:

                # change the pointer to database result in order to get unique value of the span
                if span_name == unique_span_name:
                    while offset < len(database_results[domain]['results']) and \
                        database_results[domain]['results'][offset][span_name] in used_values:
                        offset += 1
                    if offset >= len(database_results[domain]['results']):
                        return None
                    used_values.add(database_results[domain]['results'][offset][span_name])

                value = database_results[domain]['results'][offset][span_name]

                if domain == "taxi" and span_name == "type":
                    row = database_results[domain]['results'][offset]
                    add_mapping(row["color"] + ' ' + row["type"])

                elif span_name == "type":
                    affix = 's' if last_num_choices != 1 and not utterance[e:].startswith("s") and not utterance[e:].startswith("es") else ''
                    add_mapping(value + affix)

                elif span_name in ["address", "name", "departure", "destination", "food", "day"]:
                    add_mapping(titlecase(value))

                elif span_name == "postcode":
                    add_mapping(value.upper())

                elif span_name == "pricerange" and value == "moderate" and \
                    not utterance[e:].startswith(' price range') and not utterance[e:].startswith(' in price'):
                    add_mapping("moderately priced")

                else: add_mapping(value)

            # if the span name is not in database results, try to fill is with values in belief state
            elif domain in belief:
                if span_name not in belief[domain] or len(belief[domain][span_name]) == 0:
                    return None
                offset = min(offset, len(belief[domain][span_name]) - 1) 
                add_mapping(belief[domain][span_name][offset])

        # the actual lexicalization, substitute all spans with stored texts
        span_mappings.sort(key=(lambda  x: x[0]))
        lexicalized = ""
        last_s = 0
        for s, e, t in span_mappings:
            lexicalized += utterance[last_s:s] 
            lexicalized += str(t)
            last_s = e
        lexicalized += utterance[last_s:]

        return lexicalized


class LanguageModelMultiWOZ(MultiWOZTask, LanguageModelGoalTask):

    NAME = "lm_multiwoz"

    value_join_str = ', '
    pair_join_str = ': '

    @classmethod
    def value_func(cls, x): return cls.value_join_str.join(x)

    @staticmethod
    def wrap_parenthesis(x): return '' if x == '' else ' [' + x + ']'

    @classmethod
    def len_func(cls, x): return cls.pair_join_str + str(len(x))

    @classmethod
    def api_func(cls, domain, results): 
        if results['booking'] is None: booking_str = ''
        elif results['booking']: booking_str = ' booked'
        else: booking_str = ' not booked'
        return domain + booking_str, results['results']

    @classmethod
    def get_task_transforms(cls, params):
        return transforms.Compose([
            LanguageModelGoalTask.get_task_transforms(params),
            LinearizeAttributePairTransform('prev_belief', join_string=cls.pair_join_str, value_func=cls.value_func),
            LinearizeAttributeTransform('prev_belief', key_join_string=cls.value_join_str, value_join_string=cls.value_join_str, value_func=cls.wrap_parenthesis),
            LinearizeAttributePairTransform('target_belief', join_string=cls.pair_join_str, value_func=cls.value_func),
            LinearizeAttributeTransform('target_belief', key_join_string=cls.value_join_str, value_join_string=cls.value_join_str, value_func=cls.wrap_parenthesis, sort_by_key=False),
            LinearizeAttributeTransform('api_result', key_join_string=cls.value_join_str, value_join_string=None, value_func=cls.len_func, preprocess_func=cls.api_func, sort_values=False),
            PrefixAttributeTransform('target_belief', params.try_get('belief_state_prefix', None)),
            LinearizeAttributeTransform('api_call'),
            PrefixAttributeTransform('api_result', params.try_get('database_prefix', None))
        ])

    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.pairs_re = re.compile(r"(\w[\w ]*\w)" + self.pair_join_str + r"([\w: ']+)", re.IGNORECASE)
        self.domain_re = re.compile(r"(\w+\s*\w+)(?:,|\s*\[\s*([\w,:'" + self.pair_join_str + self.value_join_str + r"]*)\s*\]|$)", re.IGNORECASE)

    def _decode_query_batch(self, batch):

        batch = super()._decode_query_batch(batch)

        beliefs, active_domains = [], []
        for belief_string in batch:
            belief, active_domain = {}, None

            for i, match in enumerate(self.domain_re.finditer(belief_string)):
                
                domain, slot_value_string = match.group(1), match.group(2)
                belief[domain] = {}

                if slot_value_string is None:
                    continue

                for pair_match in self.pairs_re.finditer(slot_value_string):
                    slot, value = pair_match.group(1), pair_match.group(2)
                    belief[domain][slot] = [value]

                if i == 0:
                    active_domain = domain

            beliefs.append(belief)
            active_domains.append(active_domain)

        return beliefs, active_domains


class DoubleLanguageModelMultiWOZ(MultiWOZTask, DoubleLanguageModelGoalTask):
    
    NAME = "double_lm_multiwoz"

    value_join_str = ', '
    pair_join_str = ': '

    @classmethod
    def value_func(cls, x): return cls.value_join_str.join(x)

    @staticmethod
    def wrap_parenthesis(x): return '' if x == '' else ' [' + x + ']'

    @classmethod
    def len_func(cls, x): 
        return cls.pair_join_str + str(len(x))

    @classmethod
    def api_func(cls, domain, results): 
        if results['booking'] is None: booking_str = ''
        elif results['booking']: booking_str = ' booked'
        else: booking_str = ' not booked'
        #booking_str = ''
        return domain + booking_str, results['results']

    @classmethod
    def get_task_transforms(cls, params):
        return transforms.Compose([
            DoubleLanguageModelGoalTask.get_task_transforms(params),
            LinearizeAttributePairTransform('prev_belief', join_string=' ', value_func=cls.value_func),
            LinearizeAttributeTransform('prev_belief', key_join_string=' ', value_join_string=' ', value_func=cls.wrap_parenthesis),
            LinearizeAttributePairTransform('target_belief', join_string=cls.pair_join_str, value_func=cls.value_func),
            LinearizeAttributeTransform('target_belief', key_join_string=cls.value_join_str, value_join_string=cls.value_join_str, value_func=cls.wrap_parenthesis, sort_by_key=False),
            LinearizeAttributeTransform('api_call'),
            #ApiResultsToTokenTransform('api_result'),
            LinearizeAttributeTransform('api_result', key_join_string=cls.value_join_str, value_join_string=None, value_func=cls.len_func, preprocess_func=cls.api_func, sort_values=False)
        ])

    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.pairs_re = re.compile(r"(\w[\w ]*\w)" + self.pair_join_str + r"([\w: ']+)", re.IGNORECASE)
        self.domain_re = re.compile(r"(\w+\s*\w+)(?:,|\s*\[\s*([\w,:'" + self.pair_join_str + self.value_join_str + r"]*)\s*\]|$)", re.IGNORECASE)

    def _decode_query_batch(self, batch):

        batch = super()._decode_query_batch(batch)

        beliefs, active_domains = [], []
        for belief_string in batch:
            belief, active_domain = {}, None

            for i, match in enumerate(self.domain_re.finditer(belief_string)):
                
                domain, slot_value_string = match.group(1), match.group(2)
                belief[domain] = {}

                if slot_value_string is None:
                    continue

                for pair_match in self.pairs_re.finditer(slot_value_string):
                    slot, value = pair_match.group(1), pair_match.group(2)
                    belief[domain][slot] = [value]

                if i == 0:
                    active_domain = domain

            beliefs.append(belief)
            active_domains.append(active_domain)

        return beliefs, active_domains


class PolicyOptimizationLanguageModelMultiWOZ(MultiWOZTask, PolicyOptimizationLanguageModelGoalTask):
    
    NAME = "policy_lm_multiwoz"

    value_join_str = ', '
    pair_join_str = ': '

    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)

    @classmethod
    def value_func(cls, x): return cls.value_join_str.join(x)

    @staticmethod
    def wrap_parenthesis(x): return '' if x == '' else ' [' + x + ']'

    @classmethod
    def len_func(cls, x): 
        return cls.pair_join_str + str(len(x))

    @classmethod
    def api_func(cls, domain, results): 
        if results['booking'] is None: booking_str = ''
        elif results['booking']: booking_str = ' booked'
        else: booking_str = ' not booked'
        #booking_str = ''
        return domain + booking_str, results['results']

    @classmethod
    def get_task_transforms(cls, params):
        return transforms.Compose([
            PolicyOptimizationLanguageModelGoalTask.get_task_transforms(params),
            LinearizeAttributePairTransform('prev_belief', join_string=' ', value_func=cls.value_func),
            LinearizeAttributeTransform('prev_belief', key_join_string=' ', value_join_string=' ', value_func=cls.wrap_parenthesis),
            LinearizeAttributeTransform('api_result', key_join_string=cls.value_join_str, value_join_string=None, value_func=cls.len_func, preprocess_func=cls.api_func, sort_values=False),
            PrefixAttributeTransform('api_result', params.try_get('database_prefix', None)),
            PrefixAttributeTransform('prev_belief', params.try_get('belief_state_prefix', None))
        ])
