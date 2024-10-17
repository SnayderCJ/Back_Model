from django.contrib import admin

# Register your models here.

from .models import Training, Sample, EvaluationResult

admin.site.register(Training)
admin.site.register(Sample)
admin.site.register(EvaluationResult)