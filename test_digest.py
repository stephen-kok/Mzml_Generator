import re
sequence = "AGRPEPTIDEKSEQUENCE"
rule = r'(?<=[KR])'

print("Sequence:", sequence)
print("Rule:", rule)

matches = list(re.finditer(rule, sequence))
print("Matches:", [m.end() for m in matches])
