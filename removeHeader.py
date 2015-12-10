def getfeature(data):
    f = open(data,"r")
    for lines in f:
        v = f.next()
        yield lines.rstrip("\n"),v
    f.close()
    

import sys

data = sys.argv[1]
trueCDs = sys.argv[2]

vecs = getfeature(data)

ftrue = open(trueCDs,"w")

for h,vec in vecs:
	ftrue.write(str(vec))
	
ftrue.close()          
