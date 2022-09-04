import json
import os
import random
import copy
from fuzzywuzzy import fuzz
from aargh.utils.data import time_str_to_minutes
from aargh.data.loaders import MultiWOZ


class MultiWOZDatabase:
    """ MultiWOZ database implementation. """

    IGNORE_VALUES = {
        'hospital' : ['id'],
        'police' : ['id'],
        'attraction' : ['location', 'openhours'],
        'hotel' : ['location', 'price'],
        'restaurant' : ['location', 'introduction']
    }

    FUZZY_KEYS = {
        'hospital' : {'department'},
        'hotel' : {'name'},
        'attraction' : {'name'},
        'restaurant' : {'name', 'food'},
        'bus' : {'departure', 'destination'},
        'train' : {'departure', 'destination'},
        'police' : {'name'}
    }

    def __init__(self):
        self.data, self.data_keys = self._load_data()

    def _load_data(self):
        database_data = {}
        database_keys = {}
        
        loader = MultiWOZ()
        for domain in loader.DOMAINS:
            with open(os.path.join(loader.get_root(), f"{domain}_db.json"), "r") as f:
                for l in f:
                    if not l.startswith('##') and l.strip() != "":
                        f.seek(0)
                        break
                database_data[domain] = json.load(f)
            
            if domain in self.IGNORE_VALUES:
                for i in database_data[domain]:
                    for ignore in self.IGNORE_VALUES[domain]:
                        if ignore in i:
                            i.pop(ignore)

            database_keys[domain] = set()
            if domain == 'taxi':
                database_data[domain] =  {k.lower(): v for k, v in database_data[domain].items()}
                database_keys[domain].update([k.lower() for k in database_data[domain].keys()])
            else:
                for i, database_item in enumerate(database_data[domain]):
                    database_data[domain][i] =  {k.lower(): v for k, v in database_item.items()}
                    database_keys[domain].update([k.lower() for k in database_item.keys()])

        return database_data, database_keys

    def query(self, domain, constraints, fuzzy_ratio=90):
        """
        Returns the list of entities (dictionaries) for a given domain based on the annotation of the belief state.
        
        Arguments:
            domain (str): Name of the queried domain.
            constraints (dict[dict]): Hard constraints to the query results.
        """

        if domain == 'taxi':
            c, t, p = None, None, None
            
            c = constraints.get('color', [random.choice(self.data[domain]['taxi_colors'])])[0]
            t = constraints.get('type', [random.choice(self.data[domain]['taxi_types'])])[0]
            p = constraints.get('phone', [''.join([str(random.randint(1, 9)) for _ in range(11)])])[0]

            return [{'color': c, 'type' : t, 'phone' : p}]

        elif domain == 'hospital':

            hospital = {
                'hospital phone': '01223245151',
                'address': 'Hills Rd, Cambridge',
                'postcode': 'CB20QQ',
                'name': 'Addenbrookes'
            }

            departments = [x.strip().lower() for x in constraints.get('department', [])]
            phones = [x.strip().lower() for x in constraints.get('phone', [])]

            if len(departments) == 0 and len(phones) == 0:
                return [dict(hospital)]
            else:      
                results = []
                for i in self.data[domain]:
                    if 'department' in self.FUZZY_KEYS[domain]:
                        f = (lambda x: fuzz.partial_ratio(i['department'].lower(), x) > fuzzy_ratio)
                    else:
                        f = (lambda x: i['department'].lower() != x)

                    if any(f(x) for x in departments) and \
                       (len(phones) == 0 or any(i['phone'] == p.strip() for p in phones)):
                        results.append(dict(i))
                        results[-1].update(hospital)

                return results
    
        else:
            # Hotel database keys:      address, area, name, phone, postcode, pricerange, type, internet, parking, stars, takesbookings (other are ignored)
            # Attraction database keys: address, area, name, phone, postcode, pricerange, type, entrance fee (other are ignored)
            # Restaurant database keys: address, area, name, phone, postcode, pricerange, type, food 

            # Bus and train database contains keys: arriveby, departure, day, leaveat, destination, trainid, price, duration
            # The keys arriveby, leaveat expect a time format such as 8:45 for 8:45 am

            results = []
            query = {}

            if domain == 'attraction' and 'entrancefee' in constraints:
                constraints['entrance fee'] = constraints.pop(['entrancefee'])

            for key in self.data_keys[domain]:         
                query[key] = constraints.get(key, [])
                if len(query[key]) > 0 and key in ['arriveby', 'leaveat']:
                    query[key] = [time_str_to_minutes(x) for x in query[key]]

            for i, item in enumerate(self.data[domain]):
                for k, v in query.items():
                    if len(v) == 0 or item[k] == '?':
                        continue

                    if k == 'arriveby':
                        time = time_str_to_minutes(item[k]) 
                        if all(time > x for x in v):
                            break
                    elif k == 'leaveat':
                        time = time_str_to_minutes(item[k]) 
                        if all(time < x for x in v):
                            break
                    else:
                        if k in self.FUZZY_KEYS.get(domain, {}):
                            f = (lambda x: fuzz.partial_ratio(item[k], x) < fuzzy_ratio)
                        else:
                            f = (lambda x: item[k] != x)
                        if all(f(x.strip().lower()) for x in v):
                            break
                else:
                    result = copy.deepcopy(item)

                    if domain in ['train', 'hotel', 'restaurant']:
                        ref = constraints.get('ref', [])
                        result['ref'] = '{0:08d}'.format(i) if len(ref) == 0 else ref 

                    results.append(result)
            
            if domain == 'attraction':
               for result in results: 
                   result['entrancefee'] = result.pop('entrance fee')

            return results
