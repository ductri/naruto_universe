import logging
import re
import json

import pandas as pd

from naruto_skills.network import get


def build_url(domain, topic, list_filters=(), fields=('*',)):
    """
    :param domain: str
    :param topic: str
    :param list_filters: tuple,
    suggestion: (
        'q=*:*',
        'fq=-is_ignore:1',
        'fq=-is_noisy:1',
        'fq=is_approved:1',
        'wt=json',
        'fq=search_text:*',
        'fq=copied_at:[%sZ TO %sZ]' % (start, end), # e.g format '2018-11-07T00:00:00'
    )
    :param fields: tuple
    :param domain: ask secretary
    :return:
    """
    fields_str = 'fl=' + ','.join([f.strip() for f in fields])

    filters_str = '&'.join(list_filters)

    url = '%s/solr/topic_%s/select?%s&%s' % (domain, topic, fields_str, filters_str)
    return url


def crawl_topic(domain, topic, filters=(), limit=1e9, batch_size=5000, output_pandas=True, fields=('*',), username='', password=''):
    """

    :param domain: str
    :param topic: str ID, e.g topic = '34498'
    :param filters: tuple,
    suggestion: (
        'q=*:*',
        'fq=-is_ignore:1',
        'fq=-is_noisy:1',
        'fq=is_approved:1',
        'wt=json',
        'fq=search_text:*',
        'fq=copied_at:[%sZ TO %sZ]' % (start, end), # e.g format '2018-11-07T00:00:00'
    )
    :param limit: maximum number of records
    :param batch_size:
    :param output_pandas: output a DataFrame or not
    :param fields:
    :return:
    """
    pos = 0
    n_rows = batch_size
    data_len = 0
    data = []
    page_index = 1
    filters = tuple([f for f in filters if f[:5] != 'rows='])

    while data_len < limit:
        filters_batch = filters + ('rows=%s' % n_rows, 'start=%s' % pos)
        url = build_url(domain, topic, filters_batch, fields)
        result = get(url, username=username, password=password, result_type='text')
        try:
            result = json.loads(result)
        except json.decoder.JSONDecodeError as e:
            logging.error('Can not parse result to json. Raw result: %s', result)
            raise e
        logging.info('Crawled topic {} on page {}, {}/{} done'.format(topic, page_index,
                                                                      min(pos + n_rows, result['response']['numFound']),
                                                                      result['response']['numFound']))
        page_index += 1
        for item in result['response']['docs']:
            item['topic_id'] = topic
            data.append(item)
        pos += n_rows
        data_len += len(result['response']['docs'])
        if len(result['response']['docs']) == 0:
            break
    if output_pandas:
        df_data = pd.DataFrame(data)
        if fields != ('*', ):
            df_data = fill_fields(df_data, fields+('topic_id', ))
        return df_data
    return data


def fill_fields(df, fields, default_value=''):
    """
    Add missing field with default values, and ensure columns in the same order of `fields`
    :param df:
    :param fields: list
    :param default_value:
    :return:
    """
    for f in fields:
        if f not in df.columns:
            df[f] = default_value
    return df[list(fields)]


def remove_vietnamese_accent(utf8_str):
    INTAB = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"

    OUTTAB = "a" * 17 + "o" * 17 + "e" * 11 + "u" * 11 + "i" * 5 + "y" * 5 + "d" + \
             "A" * 17 + "O" * 17 + "E" * 11 + "U" * 11 + "I" * 5 + "Y" * 5 + "D"

    r = re.compile("|".join(INTAB))
    replaces_dict = dict(zip(INTAB, OUTTAB))

    return r.sub(lambda m: replaces_dict[m.group(0)], utf8_str)


def reformat(df_):
    df_['period'] = df_['end'].map(lambda x: str(x)[:-9])
    groups = df_.groupby('topic_id')
    new_data_acc = {}
    new_data_total_marked = {}
    new_data_total = {}
    new_index = []
    for name, gr in groups:
        gr = gr.sort_values('period')
        new_data_acc[name] = list(gr['acc'])
        new_data_total_marked[name] = list(gr['total_marked'])
        new_data_total[name] = list(gr['total'])
        new_index = list(gr['period'])

    df_acc_ = pd.DataFrame(new_data_acc)
    df_acc_.index = new_index

    df_total_marked_ = pd.DataFrame(new_data_total_marked)
    df_total_marked_.index = new_index

    df_total_ = pd.DataFrame(new_data_total)
    df_total_.index = new_index
    return df_acc_, df_total_marked_, df_total_


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(build_url(domain='http://xxxxxx.xx.x', topic='2540', list_filters=(
        'q=*:*',
        'fq=-is_ignore:1',
        'fq=-is_noisy:1',
        'wt=json',
        'fq=search_text:*',
        'fq=copied_at:[%sZ TO %sZ]' % ('2018-11-07T00:00:00', '2018-11-18T00:00:00'), # e.g format '2018-11-07T00:00:00'
    ), fields=('id', 'tags', 'created_date')))

    df = crawl_topic(domain='http://xxxxxx.xx.x', topic='4637', fields=('*',), filters=('q=*:*',
        'fq=-is_ignore:1',
        'fq=-is_noisy:1',
        'wt=json',
        'fq=search_text:*',
        'fq=copied_at:[%sZ TO %sZ]' % ('2019-02-01T00:00:00', '2019-02-15T00:00:00')),
                     batch_size=500, username='x', password='y')
    df.to_csv('temp/temp.csv', index=None)
