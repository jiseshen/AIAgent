import random

class Fact:
    def __init__(self, identifier, description, depth=0, path=None):
        self.identifier = identifier
        self.description = description
        self.depth = depth
        if path is None:
            self.path = []
        else:
            self.path = path.copy()

    def __repr__(self):
        return f"Fact({self.identifier}, \"{self.description}\")"


class Rule:
    def __init__(self, premises, conclusion):
        self.premises = premises  # List of Fact objects
        self.conclusion = conclusion  # A single Fact object

    def __repr__(self):
        premises_desc = " and ".join(fact.description[:-1].lower() for fact in self.premises)
        conclusion = self.conclusion.description[:-1].lower()
        templates = [
            f"If {premises_desc}, then {conclusion}.",
            f"When {premises_desc}, we can conclude {conclusion}.",
            f"Given that {premises_desc}, it follows that {conclusion}.",
            f"Assuming {premises_desc}, we can say {conclusion}.",
            f"Based on {premises_desc}, it is evident that {conclusion}."
        ]
        return random.choice(templates)  # Randomly select a template


class Reasoner:
    def __init__(self, facts=None, rules=None, all_facts=None):
        if facts is None:
            facts = []
        if rules is None:
            rules = []
        self.facts = {fact.identifier: fact for fact in facts}
        self.all_facts = {fact.identifier: fact for fact in all_facts}
        self.rules = rules
        self.string_context = ""
        self.string_reasoning = ""

    def add_fact(self, fact):
        self.facts[fact.identifier] = fact

    def add_rule(self, rule):
        self.rules.append(rule)

    def forward_chain(self):
        # Print and store the initial context without identifiers
        self.string_context = "Facts:\n" + "\n".join(fact.description for fact in self.facts.values())
        self.string_context += "\nRules:\n" + "\n".join(str(rule) for rule in self.rules)

        # Store reasoning steps
        reasoning_steps = []
        statement = None
        answer = None
        cur_depth = 0
        new_facts_added = True
        while new_facts_added:
            cur_depth += 1
            new_facts_added = False
            for rule in self.rules:
                premises_ids = [fact.identifier for fact in rule.premises]
                if all(pid in self.facts for pid in premises_ids):
                    if rule.conclusion.identifier not in self.facts:
                        conclusion = rule.conclusion.description[:-1].lower()

                        # Choose a random template for reasoning
                        templates = [
                            f"Applied Rule: since the fact that {str(rule)[:-1].lower()}, we can see that {conclusion}.",
                            f"Since we know that {str(rule)[:-1].lower()}, it follows that {conclusion}.",
                            f"Because {str(rule)[:-1].lower()}, we can conclude that {conclusion}.",
                            f"Given that {str(rule)[:-1].lower()}, it is evident that {conclusion}.",
                            f"Applying the logic of {str(rule)[:-1].lower()}, we deduce that {conclusion}."
                        ]
                        temp = random.choice(templates)
                        new_fact = Fact(rule.conclusion.identifier, rule.conclusion.description, depth=cur_depth, path=self.facts[premises_ids[0]].path + [temp])
                        self.add_fact(new_fact)
                        
                        reasoning_steps.append(random.choice(templates))
                        new_facts_added = True


        # Randomly choose a statement from the derived facts
        if random.random() < 0.5:
            selected_fact = sorted(list(self.facts.values()), key=lambda x: x.depth, reverse=True)[0]
            depth = selected_fact.depth
            statement = selected_fact.description
            reasoning_steps = selected_fact.path
            answer = "True"
        else:
            answer = "False"
            depth = cur_depth
            statement = "Chuan is handsome"
            facts_descriptions = [fact.description for fact in self.facts.values()]
            for fact in self.all_facts.values():
                if fact.description not in facts_descriptions:
                    statement = fact.description
                    break

        # Prepare the final output string
        output = {"context": self.string_context, 
                  "statement": statement, 
                  "label": answer, 
                  "reasoning": reasoning_steps if reasoning_steps else "No new facts were derived.", 
                  "depth": depth}

        return output

    def get_facts(self):
        return list(self.facts.values())

    def get_a_false_statement(self, rules, facts):
        conclusions = [rule.conclusion.description for rule in rules]
        facts = [fact.description for fact in facts.values()]
        for conclusion in conclusions:
            if conclusion not in facts:
                return conclusion
        return facts[0]


def string_from_facts_and_rules(facts, rules, all_facts):
    reasoner = Reasoner(facts, rules, all_facts)
    return reasoner.forward_chain()


# facts = [
#     Fact("A", "the sky is blue"),
#     Fact("B", "grass is green")
# ]
#
# # Define rules with Fact objects as premises and conclusion
# rules = [
#     Rule([facts[0], facts[1]], Fact("C", "the sky and grass are both visible")),
#     Rule([Fact("C", "the sky and grass are both visible")], Fact("D", "it is daytime")),
#     Rule([Fact("D", "it is daytime"), facts[0]], Fact("E", "the sun is shining"))
# ]
#
# print(string_from_facts_and_rules(facts, rules))