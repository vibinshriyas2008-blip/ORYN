import urllib.request, json

BASE = 'http://localhost:8000'

def post(path, data):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'}
    )
    r = urllib.request.urlopen(req)
    return json.loads(r.read())

def get(path):
    r = urllib.request.urlopen(BASE + path)
    return json.loads(r.read())

# Test signup
u = post('/auth/signup', {'name': 'ORYN Test', 'email': 'test@oryn.ai', 'password': 'test123'})
uid = u['user_id']
print('Signup OK: user_id=' + str(uid))

# Test onboarding
ob = post('/onboarding', {'user_id': uid, 'genres': ['Sci-Fi', 'Drama'], 'liked_movie_ids': [1, 2, 3], 'ratings': {}})
print('Onboarding: ' + ob['status'])

# Test recommendations (cold user)
recs = post('/recommend', {'user_id': uid, 'n': 5, 'diversity': True})
print('Cold recommendations: ' + str(len(recs)) + ' items')
for r in recs[:3]:
    print('  [' + r['title'] + '] score=' + str(r['score']) + ' | ' + r['explanation'])

# Test popular
pop = get('/popular?n=3')
print('Popular: ' + str([m['title'] for m in pop[:3]]))

# Test metrics
metrics = get('/metrics')
print('Metrics keys: ' + str(list(metrics.keys())))
print('Hybrid P@5: ' + str(metrics['overall']['Hybrid']['P@5']))
print('Hybrid+Rerank NDCG@10: ' + str(metrics['overall']['Hybrid+Rerank']['NDCG@10']))

print('\nALL TESTS PASSED')
