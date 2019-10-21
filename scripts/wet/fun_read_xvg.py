def read_xvg_energy(filename):
    data = {}
    with open(filename) as f:
        s_tmp = ""
        s = f.readline()
        while s.startswith("-") is not True:
            s_tmp = s
            s = f.readline()
        attrs = s_tmp.strip().split()  # Attributes of columns
        s = f.readline()
        while len(s) > 0:
            # print(s)
            name = ""
            i = 0
            s = s.split()
            while not s[i][0].isdecimal() and not s[i][0] == "-":
                name += s[i]
                i += 1
            d_dic = {}
            for att in attrs[1:]:
                d_dic[att] = float(s[i])
                i += 1
            d_dic["Unit"] = s[-1]
            data[name] = d_dic
            s = f.readline()
    return data
