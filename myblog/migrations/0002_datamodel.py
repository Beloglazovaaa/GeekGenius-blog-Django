# Generated by Django 3.2.25 on 2024-04-09 14:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myblog', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='DataModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('feature1', models.FloatField()),
                ('feature2', models.FloatField()),
                ('target', models.FloatField()),
            ],
        ),
    ]