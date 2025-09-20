from rest_framework import serializers


class RecommendationSerializer(serializers.Serializer):
    movie_id = serializers.IntegerField()
    title = serializers.CharField()


