from django.shortcuts import render
from .forms import *

def michelson_interferometer(request):
    delta_L = 0  # Initialize optical path difference variable
    
    if request.method == 'POST':
        form = MichelsonInterferometerForm(request.POST)
        if form.is_valid():
            # Retrieve the compensating plate parameters
            plate_thickness = form.cleaned_data.get('plate_thickness')
            refractive_real = form.cleaned_data.get('plate_refractive_real')
            refractive_imag = form.cleaned_data.get('plate_refractive_imag')
            
            # If refractive index values are provided, create a complex refractive index
            if plate_thickness and refractive_real is not None and refractive_imag is not None:
                refractive_index = complex(refractive_real, refractive_imag)
                # Calculate the optical path difference using the complex refractive index
                delta_L = plate_thickness * (refractive_index.real - 1)
                # Optionally, consider the effect of the imaginary part (absorption) in calculations
                # For simplicity, we omit that part here, but you can include it if you need to model attenuation

            return render(request, 'interferometers/michelson.html', {'form': form, 'delta_L': delta_L})
    else:
        form = MichelsonInterferometerForm()

    return render(request, 'interferometers/michelson.html', {'form': form})


def fabry_perot_interferometer(request):
    return render(request, 'interferometers/fabry_perot.html')

def mach_zehnder_interferometer(request):
    return render(request, 'interferometers/mach_zehnder.html')
