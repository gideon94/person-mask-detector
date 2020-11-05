from django.test import TestCase 
from rest_framework import status
from django.urls import reverse
from rest_framework.test import APIClient
import json
import os

from django.conf import settings

MEDIA_ROOT = getattr(settings, "MEDIA_ROOT", None) 

client = APIClient()
# Create your tests here.
class InputValidationTestCase(TestCase):
    def test_create_valid(self):
        response = client.post(
            reverse('predict'),
            {
                "input_img": open(os.path.join( MEDIA_ROOT, "yoda.jpg"),"rb")
            }
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_create_in_valid_file_type(self):
        response = client.post(
            reverse('predict'),
            {
                "input_img": open(os.path.join( MEDIA_ROOT, "test.txt"),"rb")
            }
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_create_empty_file(self):
        response = client.post(
            reverse('predict'),
            {

            },
        )
        print(response.status_code)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
