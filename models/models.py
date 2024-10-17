from django.db import models

class Training(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    model_file = models.FileField(upload_to='models/')
    accuracy = models.FloatField(null=True, blank=True)
    loss = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f'Training {self.id} - Accuracy: {self.accuracy}'

class Sample(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    frames_directory = models.CharField(max_length=255)
    normalized_frames_directory = models.CharField(max_length=255, blank=True, null=True)
    keypoints_file = models.FileField(upload_to='keypoints/', blank=True, null=True)
    label = models.CharField(max_length=100)

    def __str__(self):
        return f'Sample {self.id} - Label: {self.label}'

class EvaluationResult(models.Model):
    sample = models.ForeignKey(Sample, on_delete=models.CASCADE)
    prediction = models.CharField(max_length=100)
    confidence = models.FloatField()

    def __str__(self):
        return f'Evaluation {self.id} - Prediction: {self.prediction}'
