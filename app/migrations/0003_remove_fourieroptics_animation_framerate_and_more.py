# Generated by Django 5.1.5 on 2025-01-28 19:32

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_alter_fourieroptics_intensity_animation_path_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='fourieroptics',
            name='animation_framerate',
        ),
        migrations.RemoveField(
            model_name='fourieroptics',
            name='intensity_animation_path',
        ),
        migrations.RemoveField(
            model_name='fourieroptics',
            name='rgb_animation_path',
        ),
    ]
