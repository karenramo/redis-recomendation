import time
import threading
import json
import math
import redis
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import col, sum as spark_sum

# Inicializa findspark y configura Spark
findspark.init()

spark = SparkSession.builder \
    .appName("VideoRecommendation") \
    .getOrCreate()

# Configura la conexión a Redis
redis_host = 'localhost'  # Host de Redis (puedes ajustarlo según tu configuración)
redis_port = 6379  # Puerto de Redis (por defecto es 6379)
redis_db = 0  # Número de la base de datos de Redis a utilizar

redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db)

# Define el esquema del DataFrame vacío
schema = StructType([
    StructField("tag", StringType(), True),
    StructField("duration", FloatType(), True)
])

class VideoCounter:
    def __init__(self):
        self.counters = spark.createDataFrame([], schema=schema)
        self.total_views = 0
        self.running = True
        self.duration = 0

    def add_view(self, name, tag, duration):
        new_view = spark.createDataFrame([(tag, duration)], schema=schema)
        self.counters = self.counters.union(new_view)
        self.total_views += 1

    def start_counter(self):
        self.running = True
        start_time = time.time()
        while self.running:
            time.sleep(1)
        self.duration = time.time() - start_time

    def stop_counter(self):
        self.running = False

    def get_recommendations(self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)

        # Obtiene el perfil de usuario desde Redis
        user_profile = self.calculate_user_profile()
        print("User profile:", user_profile)

        recommendations = []
        for video in data['videos']:
            # Obtiene el perfil de video desde Redis (usando el nombre del video como clave)
            video_profile = self.calculate_video_profile(video)
            print("Video profile:", video_profile)

            similarity = self.calculate_similarity(user_profile, video_profile)
            recommendations.append((video, similarity))

        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations

    def calculate_user_profile(self):
        user_profile = self.counters.groupBy('tag').agg(spark_sum('duration').alias('total_duration')).collect()
        user_profile_dict = {row['tag']: row['total_duration'] for row in user_profile}

        # Almacena el perfil de usuario en Redis
        for tag, value in user_profile_dict.items():
            redis_client.hset('user_profile', tag, value)
        return user_profile_dict

    def calculate_video_profile(self, video):
        video_profile = {}
        # Usamos el 'tag' como el género del video
        genre = video['tag']
        video_profile[genre] = 1  # Asignamos un peso fijo para cada video basado en el tag

        # Almacena el perfil de video en Redis (usando el nombre del video como clave)
        redis_client.hset(video['name'], genre, 1)
        return video_profile

    def calculate_similarity(self, profile1, profile2):
        common_tags = set(profile1.keys()) & set(profile2.keys())
        if not common_tags:
            return 0.0
        dot_product = sum(float(profile1[tag]) * float(profile2[tag]) for tag in common_tags)
        magnitude_profile1 = math.sqrt(sum(float(value) ** 2 for value in profile1.values()))
        magnitude_profile2 = math.sqrt(sum(float(value) ** 2 for value in profile2.values()))
        similarity = dot_product / (magnitude_profile1 * magnitude_profile2)
        return similarity

    def store_recommendations_in_redis(self, recommendations):
        # Guarda las recomendaciones en Redis usando un hash por usuario
        for i, (video, similarity) in enumerate(recommendations, start=1):
            redis_client.hset('recommendations', f'rec{i}', json.dumps(video))


def main():
    video_counter = VideoCounter()
    video_count = 0

    while video_count < 3:
        name = input("Ingrese el nombre del video (o 'exit' para salir): ")
        if name.lower() == 'exit' and video_count < 3:
            print("Debe ingresar al menos 3 videos antes de salir.")
            continue
        elif name.lower() == 'exit':
            break
        tag = input("Ingrese el tag (género) del video: ")

        print("Iniciando contador de tiempo. Presione 's' para detener.")
        counter_thread = threading.Thread(target=video_counter.start_counter)
        counter_thread.start()

        while True:
            stop = input()
            if stop.lower() == 's':
                video_counter.stop_counter()
                counter_thread.join()
                break

        duration = video_counter.duration
        video_counter.add_view(name, tag, duration)
        print(f"Duración del video '{name}' con tag '{tag}': {duration:.2f} segundos")

        video_count += 1
        if video_count >= 3:
            # Imprimir contadores de vistas
            total_durations = {row['tag']: row['total_duration'] for row in video_counter.counters.groupBy('tag').agg(spark_sum('duration')).collect()}
            print(f"Total views: {video_counter.total_views}")
            print("Duración total por tag (género):", total_durations)

            # Obtener recomendaciones basadas en un archivo JSON
            recommendations = video_counter.get_recommendations('recommendations.json')
            print("Recommended videos based on your watching history:")
            for video, similarity in recommendations:
                print(f"- {video['name']} (tag: {video['tag']}) - Similarity: {similarity:.2f}")

            # Almacenar las recomendaciones en Redis
            video_counter.store_recommendations_in_redis(recommendations)
            break


if __name__ == "__main__":
    main()
