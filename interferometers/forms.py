from django import forms

class MichelsonInterferometerForm(forms.Form):
    wavelength = forms.FloatField(initial=632.8)  # Wavelength in nm (HeNe laser)
    intensity = forms.FloatField(initial=1.0)  # Intensity
    grid_size = forms.FloatField(initial=15)  # Grid size in mm
    resolution = forms.IntegerField(initial=500)  # Grid resolution (NxN)
    arm1_length = forms.FloatField(initial=8)  # Length of arm 1 in cm
    arm2_length = forms.FloatField(initial=7)  # Length of arm 2 in cm
    laser_to_beamsplitter = forms.FloatField(initial=3)  # Laser to beamsplitter distance (cm)
    beamsplitter_to_screen = forms.FloatField(initial=5)  # Beamsplitter to screen distance (cm)
    tilt_x = forms.FloatField(initial=0.5)  # Mirror 1 tilt (mrad)
    tilt_y = forms.FloatField(initial=0.0)  # Mirror 2 tilt (mrad)
    
    # Compensating Plate Parameters
    add_plate=forms.BooleanField(required=False,initial=False)

    plate_thickness = forms.FloatField(initial=0.1, required=False)  # Thickness of the compensating plate in cm
    plate_refractive_real = forms.FloatField(initial=1.5, required=False)  # Real part of refractive index
    plate_refractive_imag = forms.FloatField(initial=0.0, required=False)  # Imaginary part of refractive index


    # Tilt Angle (in mrad)
    plate_tilt = forms.FloatField(initial=0.0)  # Tilt angle of the compensating plate in mrad
    # Distance between Plate and Mirror 1
    plate_to_mirror1 = forms.FloatField(initial=5.0)  # Distance between plate and Mirror 1 (cm)