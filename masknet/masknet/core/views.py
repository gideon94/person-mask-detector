# from masknet.core.mask_net import get_pred_mask
# from masknet.core.person_detect import get_person
from datetime import datetime

import cv2
import numpy as np
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from rest_framework import status
from rest_framework.generics import GenericAPIView
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from masknet.core import serializers
from masknet.core.person_detect_yolo import get_persons

SAVE_IMG = getattr(settings, "SAVE_IMG", False) 


class PredictAPI(GenericAPIView):
    serializer_class = serializers.PredictSerializer
    parser_classes = ((FormParser, MultiPartParser))

    def get(self, request, *args, **kwargs):
        result = dict()
        result['status'] = True
        result['message'] = "Please use post method to get Prediction"
        return Response(result)

    def post(self, request, *args, **kwargs):
        result = dict()
        s = self.get_serializer(data=request.data)
        if s.is_valid():
            result['status'] = True
            fs = FileSystemStorage()
            start = datetime.now()
            req_file = s.validated_data['input_img']
            
            if SAVE_IMG:
                filename = fs.save(req_file.name, req_file)
                uploaded_file_url = fs.path(filename)
                result['path'] = uploaded_file_url
                img_obj = cv2.imread(uploaded_file_url)
            else:
                img_obj = cv2.imdecode(np.fromstring(req_file.read(), np.uint8), cv2.IMREAD_COLOR)
            resp = get_persons(img_obj)
            end = datetime.now()
            result['throughput'] = "{0} seconds".format((end - start).seconds)
            if len(resp) == 0:
                result['status'] = False
                result['message'] = "No Persons found!"
            else:
                result['result'] = resp
                result['message'] = "At least {0} persons found.".format(
                    len(resp))
            return Response(result)
        else:
            result['status'] = False
            result['message'] = "Invalid File"
            result['errors'] = s.errors
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
