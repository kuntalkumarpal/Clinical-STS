# this is a sample feature to use as a template

def IdenticalString(sent1, sent2, additional_data=None):
  identical = 1 if sent1.lower() == sent2.lower() else 0
  return [identical]
