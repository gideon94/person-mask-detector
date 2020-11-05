from rest_framework import serializers


class PredictSerializer(serializers.Serializer):
    input_img = serializers.ImageField(max_length=None, allow_empty_file=False)