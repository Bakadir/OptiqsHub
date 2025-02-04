from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.utils.text import slugify 


class Profile(models.Model):
   
    user = models.OneToOneField(User, on_delete=models.CASCADE)
 
    slug = models.SlugField(unique=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.slug:
            base_slug = slugify(f"{self.user.username}")
            slug = base_slug
            num = 1
            while Profile.objects.filter(slug=slug).exists():
                slug = f"{base_slug}-{num}"
                num += 1
            self.slug = slug
        super().save(*args, **kwargs)

    def __str__(self):
        return self.user.username

class FourierOptics(models.Model):
    VISIBILITY_CHOICES = [
        ('public', 'Public'),
        ('private', 'Private'),
    ]
    visibility = models.CharField(
        max_length=7,
        choices=VISIBILITY_CHOICES,
        default='private',  # Default value
    )

    title = models.CharField(max_length=255)
    created_by = models.ForeignKey(User, related_name='fourieroptics', on_delete=models.CASCADE)
    created_at = models.DateTimeField(default=timezone.now)

    aperture_type = models.CharField(max_length=100)
    wavelengths = models.JSONField()  
    intensities = models.JSONField() 

    number_of_slits = models.IntegerField()
    distance_between_slits = models.FloatField() 
    slit_width = models.FloatField() 
    slit_height = models.FloatField()  

    aperture_radius = models.FloatField()  

    add_lens = models.BooleanField()
    distance_lens_to_aperture = models.FloatField()  
    focal_length = models.FloatField()  

    distance_screen_to_aperture = models.FloatField()  
    screen_width = models.FloatField()  

    resolution = models.IntegerField()
    animation_frames = models.IntegerField()


    anim_heatmap_plot_data = models.JSONField()
    anim_3d_plot_data = models.JSONField()
    anim_lines_plot_data = models.JSONField()
    anim_rgb_plot_data = models.JSONField()

    #rgb_animation_path = models.CharField(max_length=255, null=True, blank=True)
    #intensity_animation_path = models.CharField(max_length=255, null=True, blank=True)
    """ 
    rgb_animation_path = models.FileField(upload_to='animations/', null=True, blank=True)
    intensity_animation_path = models.FileField(upload_to='animations/', null=True, blank=True)


    
    @property
    def rgb_animation_path_url(self):
        if self.rgb_animation_path and hasattr(self.rgb_animation_path, 'url'):
            return self.rgb_animation_path.url
    def intensity_animation_path_url(self):
        if self.intensity_animation_path and hasattr(self.intensity_animation_path, 'url'):
            return self.intensity_animation_path.url
         """
    def get_absolute_url(self):
        # This will generate a URL like /fourieroptics/username/title/
        return f"/{self.created_by.profile.slug}/{slugify(self.title)}/"
    def __str__(self):
        return f"Simulation {self.id}"
