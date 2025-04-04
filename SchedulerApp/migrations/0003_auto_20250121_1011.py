# Generated by Django 3.2.25 on 2025-01-21 04:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SchedulerApp', '0002_section_num_class_in_week'),
    ]

    operations = [
        migrations.AddField(
            model_name='course',
            name='classes_per_week',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='course',
            name='max_numb_students',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='meetingtime',
            name='day',
            field=models.CharField(choices=[('Monday', 'Monday'), ('Tuesday', 'Tuesday'), ('Wednesday', 'Wednesday'), ('Thursday', 'Thursday'), ('Friday', 'Friday')], max_length=15),
        ),
        migrations.AlterField(
            model_name='meetingtime',
            name='time',
            field=models.CharField(choices=[('8:00 - 9:00', '8:00 - 9:00'), ('9:00 - 10:00', '9:00 - 10:00'), ('10:00 - 11:00', '10:00 - 11:00'), ('11:00 - 12:00', '11:00 - 12:00'), ('12:00 - 1:00', '12:00 - 1:00'), ('1:00 - 2:00', '1:00 - 2:00'), ('2:00 - 3:00', '2:00 - 3:00'), ('3:00 - 4:00', '3:00 - 4:00'), ('4:00 - 5:00', '4:00 - 5:00')], default='11:30 - 12:30', max_length=50),
        ),
    ]
