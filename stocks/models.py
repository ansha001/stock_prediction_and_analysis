from django.db import models

class Stock(models.Model):
    code = models.CharField(max_length=10)
    name = models.CharField(max_length=100)