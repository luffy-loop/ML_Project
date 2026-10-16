from fastapi.testclient import TestClient
from api.main import app
client=TestClient(app)
def test_health_reports_real_artifact_state():
 r=client.get('/api/health');assert r.status_code==200;body=r.json();assert body['status']=='ok';assert isinstance(body['model_available'],bool);assert 'validation_note' in body
def test_metrics_has_explicit_empty_state():
 r=client.get('/api/metrics');assert r.status_code==200;body=r.json();assert isinstance(body['available'],bool);assert body['available'] or 'message' in body
def test_prediction_rejects_non_csv_file():
 r=client.post('/api/predict',files={'file':('notes.txt',b'x,y\n1,2\n','text/plain')});assert r.status_code==400
def test_prediction_rejects_empty_payload():
 r=client.post('/api/predict',files={'file':('empty.csv',b'','text/csv')});assert r.status_code==400
