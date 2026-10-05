import json, sys


def role_key(site: str) -> str:
    name, _, rows = site.partition("|")
    for a, b in (("_row@", "@"), ("_mma@", "@"), ("q4_gdn_pipe@", "q4_gdn@"), ("q4_gdn_step@", "q4_gdn@")):
        name = name.replace(a, b)
    if "#[" in name:
        head, shape = name.split("#[", 1)
        name = head + "#[" + ", ".join(shape.rstrip("]").split(", ")[1:]) + "]"
    return f"{name}|{int(rows or 1)}"


p = json.load(open(sys.argv[1]))
roles = {}
for site, v in p["sites"].items():
    k = role_key(site)
    assert k not in roles or roles[k] == v, k
    roles[k] = v
p["roles"] = roles
json.dump(p, open(sys.argv[1], "w"), indent=1)
print(len(roles), "roles;", sorted({k.split("|")[0] for k in roles}))
