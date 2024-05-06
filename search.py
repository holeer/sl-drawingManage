from elasticsearch import Elasticsearch
import json
import jieba.analyse as analyse


es = Elasticsearch()
mapping = {
    'properties': {
        'title': {
            'type': 'text',
            'analyzer': 'ik_max_word',
            'search_analyzer': 'ik_max_word'
        }
    }
}


def init(index_name):
    es.indices.delete(index=index_name, ignore=[400, 404])
    es.indices.create(index=index_name, ignore=400)
    result = es.indices.put_mapping(index=index_name, doc_type='politics', body=mapping)
    return result


def add(index_name, add_data):
    result = es.index(index=index_name, doc_type='politics', body=add_data)
    return result


# 搜索优先级：
# 1.标签、文件名
# 2.项目名、图名、二级图名
# 3.注释
# 4.相关人员、图号
# 5.设计说明书
def search(index_name, keyword, max_size=760):
    pattern = {
        'query': {
            'bool': {
                'must': [
                    {
                        'range': {
                            'drawing_name': {
                                'gt': 0
                            }
                        }
                    },
                    {
                        'boosting': {
                            'positive': {
                                'multi_match': {
                                    'query': keyword,
                                    'fields': ['drawing_name^4', 'project_name^4', 'designer^2', 'viewer^2', 'reviewer^2', 'sketch^4', 'sheet^4', 'label^5', 'annotation^3', 'file_name^5', 'drawing_num^2'],
                                    'fuzziness': 'AUTO'
                                }
                            },
                            'negative': {
                                'multi_match': {
                                    'fields': ['label'],
                                    'query': '说明文档',
                                    'fuzziness': 'AUTO'
                                }
                            },
                            'negative_boost': 0.05
                        }
                    }
                ]
            }
        },
        'size': max_size
    }
    results = es.search(index=index_name, doc_type='politics', body=pattern)
    re_score = []
    for result in results['hits']['hits']:
        if '说明文档' in result['_source']['label']:
            result['_score'] = 0.0
        re_score.append(result)
    re_score = sorted(re_score, key=lambda y: y['_score'], reverse=True)
    results['hits']['hits'] = re_score
    results['hits']['max_score'] = re_score[0]['_score']
    return json.dumps(results, indent=2, ensure_ascii=False)


def wordcloud(index_name, max_size=760):
    content_list = []
    query = es.search(index=index_name, doc_type='politics', size=max_size)
    results = query['hits']['hits']
    for result in results:
        for k, v in result['_source'].items():
            content_list.append(str(v))

    content = ''.join(content_list).replace(' ', '')
    for ch in '!"#$%&()*+,-./:;<=>?@[\\]^_‘{|}~':
        content = content.replace(ch, '')
    # 修改allowPOS来限定返回关键词的词性
    out = analyse.extract_tags(content, topK=50, withWeight=True, allowPOS=('n', 'ns', 'nr', 'vn'))
    return out
