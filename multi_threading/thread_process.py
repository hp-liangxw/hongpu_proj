import logging
import requests
import json
from multi_threading.json_coders import HPJsonEncoder


def _send_image_threaded_impl(json_data):
    headers = {'content-type': "application/json"}
    url = 'http://{}:{}/vi_detect'.format("52.130.81.14", "8001")
    print('send the image to {}'.format(url))
    response = requests.post(url, data=json.dumps(json_data, cls=HPJsonEncoder), headers=headers, timeout=30)

    if response.status_code == 200:
        res_data = response.json()
        logging.info('get vi_detect result: %s', res_data)
    else:
        logging.error('error')


with open(r'test.json', 'r') as f:
    json_str = json.load(f)

print(json_str)


_send_image_threaded_impl(json_str)