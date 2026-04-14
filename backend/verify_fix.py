import urllib.request, json

def test_debug(msg):
    data = json.dumps({"message": msg}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/debug", data=data, headers={"Content-Type": "application/json"})
    try:
        res = urllib.request.urlopen(req).read().decode()
        return json.loads(res).get("raw_extracted", {})
    except Exception as e:
        return str(e)

print("Test 1: 'spend 500 on shoes'")
print(test_debug("spend 500 on shoes"))

print("\nTest 2: 'average daily spend'")
print(test_debug("average daily spend"))

print("\nTest 3: 'total expense this month'")
print(test_debug("total expense this month"))
