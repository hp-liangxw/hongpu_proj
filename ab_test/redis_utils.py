'''redis.'''
import os
import redis
redis_host = os.getenv('REDIS_HOST', 'redis')

REDIS_CACHE = redis.Redis(host=redis_host, db=1)  # redis 缓存
REDIS_MQ = redis.Redis(host=redis_host, db=2, charset="utf-8", decode_responses=True)  # redis 消息队列
REDIS_CONFIG = redis.Redis(host=redis_host, db=3, charset="utf-8", decode_responses=True)  # redis 服务配置
