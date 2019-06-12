import requests
import logging
import urllib
from requests.auth import HTTPBasicAuth


def get(url, timeout=10, number_time_retry=5, result_type='json', username='', password=''):
    """
    Send HTTP GET request with timeout constraint. If timeout exception is raised, it will
    retry as most number_time_retry times.
    :param url:
    :param timeout: default 10 s
    :param number_time_retry: default 2 times
    :param result_type: type of returned result, default json
    :param username:
    :param password:
    :return:
    Return result in dict format.
    Return None if exceed number_time_retry
    """
    count_time_retry = 0
    while True:
        try:
            if (username == '') or (password == ''):
                auth = None
            else:
                auth = HTTPBasicAuth(username, password)
            logging.debug('Sending GET request with URL: %s', url)
            result = requests.get(url, timeout=timeout, auth=auth)
            break
        except requests.exceptions.RequestException as e:
            logging.warning('Timeout expcetion: ' + str(e))
            logging.warning('Retry ' + str(number_time_retry) + '-th')
            count_time_retry += 1
            if count_time_retry == number_time_retry:
                logging.exception('Exceed number of retries: %s', number_time_retry)
                raise e

    if result_type == 'json':
        return result.json()
    else:
        return result.text


def post(url, payload_json, timeout=10, number_time_retry=100, result_type='json', proxy=None, username='', password=''):
    """
    Send HTTP POST request with timeout constraint. If timeout exception is raised, it will
    retry as most number_time_retry times.
    The function returns None for unsuccessful request because of network
    :param url:
    :param payload_json:
    :param timeout:
    :param number_time_retry:
    :param result_type:
    :param username:
    :param password:
    :return:
    """
    count_time_retry = 0
    while True:
        result = None
        try:
            if proxy:
                if "http" not in proxy and "@" in proxy:
                    part1, part2 = proxy.split("@")
                    user_name, password = part1.split(":")
                    proxy = "http://{}:{}@{}".format(urllib.parse.quote(user_name),
                            urllib.parse.quote(password), part2)
                elif "http" not in proxy:
                    proxy = "http://" + proxy

                proxies = {'https': proxy}
                result = requests.post(url, json=payload_json, timeout=timeout, proxies=proxies)
            else:
                if (username == '') or (password == ''):
                    auth = None
                else:
                    auth = HTTPBasicAuth(username, password)
                result = requests.post(url, json=payload_json, timeout=timeout, auth=auth)
            if result_type == 'json':
                return result.json()
            else:
                return result.text
        finally:
            count_time_retry += 1
            if count_time_retry == number_time_retry:
                raise ExceedNetworkRetryLimitException(number_time_retry)


class ExceedNetworkRetryLimitException(Exception):
    def __init__(self, max_no_retry):
        Exception.__init__(self)
        self.max_no_retry = max_no_retry

    def __str__(self):
        return 'Maximum number of re-trying: {}'.format(self.max_no_retry)
