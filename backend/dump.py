import urllib.request, json
req = urllib.request.Request('http://127.0.0.1:8000/debug', data=json.dumps({'message': 'spend 500 on shoes'}).encode(), headers={'Content-Type': 'application/json'})
with open('out.json', 'w') as f:
    f.write(urllib.request.urlopen(req).read().decode())
