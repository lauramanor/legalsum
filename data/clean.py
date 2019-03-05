import json





if __name__ == '__main__':

    with open('tldrlegal.json') as f:
        data = json.load(f)

    data_tldrlegal = []
    for document in data:
        for info in document.values():
            data_tldrlegal.append(info)

    pass


    with open('tosdr.json') as f:
        data = json.load(f)

    data_tosdr = []
    for document in data:
        for info in document.values():
            data_tosdr.append(info)

    pass