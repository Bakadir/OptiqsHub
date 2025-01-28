import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import plotly.graph_objects as go
import plotly.io as pio
from scipy.fft import fft2, ifft2, fftfreq, fftshift
from scipy import fft
import shutil

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, FileResponse
from django import forms

from app.models import *
from .forms import *

def download_animation(request):
    file_path = 'static/fourieroptics/rgb_animation.gif'
    return FileResponse(open(file_path, 'rb'), as_attachment=True, filename='rgb_animation.gif')
def download_animation_intensity(request):
    file_path = 'static/fourieroptics/intensity_animation.gif'
    return FileResponse(open(file_path, 'rb'), as_attachment=True, filename='intensity_animation.gif')

from django.http import JsonResponse
from PIL import Image
from pathlib import Path
import time

from PIL import Image
import numpy as np
from pathlib import Path

def ApertureFromImage(image_path, Nx, Ny):
    """
    Resizes an image to fit within an aperture of a given size and inserts it into a black screen
    of size (Nx, Ny).

    Parameters:
        image_path (str): Path to the image file.
        Nx (int): Number of pixels in the x-dimension of the screen.
        Ny (int): Number of pixels in the y-dimension of the screen.
        qwt_width_mm (float): Width of the aperture in millimeters.
        qwt_height_mm (float): Height of the aperture in millimeters.
        screen_width_mm (float): Physical width of the screen in millimeters.
        screen_height_mm (float): Physical height of the screen in millimeters.

    Returns:
        np.ndarray: Grayscale aperture with the resized image centered on a black background.
    """
    # Load the image
    img = Image.open(Path(image_path))
    img = img.convert("RGB")

    # Convert the desired aperture size from millimeters to pixels
    
    # Rescale the image to the desired aperture size in pixels
    rescaled_img = img.resize((Nx, Ny), Image.Resampling.LANCZOS)
    imgRGB = np.asarray(rescaled_img) / 255.0

    # Convert the image to grayscale
    t = 0.2990 * imgRGB[:, :, 0] + 0.5870 * imgRGB[:, :, 1] + 0.1140 * imgRGB[:, :, 2]

 
    # Flip vertically if required (depends on coordinate system)
    t = np.flip(t, axis=0)

    return t


def home(request):
    

    if request.method == "POST":
        form = FourierDiff(request.POST)
        
        WavelengthIntensityFormSet = formset_factory(WavelengthIntensityForm, extra=1)
        formset = WavelengthIntensityFormSet(request.POST)
        if form.is_valid() and formset.is_valid():
            if 'add_matrix' in request.POST:
                formset = formset_factory(WavelengthIntensityForm, extra=1)(initial=[form.cleaned_data for form in formset ])
                context = {'form': form,'formset':formset}

            elif 'delete_matrix' in request.POST:
                if formset.total_form_count()>1:
                    formset = formset_factory(WavelengthIntensityForm, extra=-1)(initial=[form.cleaned_data for form in formset])
                context = {'form': form,'formset':formset}
            else :
                aperture_type = form.cleaned_data['aperture_type'] 
                wavelengths = []
                intensities = []
                

                for f in formset:
                    
                    wavelengths.append(f.cleaned_data.get('wavelength',660.0) * 1e-9)  
                    intensities.append(f.cleaned_data.get('intensity',1.0))
                    
            
                number_of_slits = form.cleaned_data['number_of_slits']
                distance_between_slits = form.cleaned_data['distance_between_slits'] * 1e-3  
                slit_width = form.cleaned_data['slit_width'] * 1e-3 
                slit_height = form.cleaned_data['slit_height'] * 1e-3 


                aperture_radius = form.cleaned_data['aperture_radius'] * 1e-3  

                add_lens = form.cleaned_data['add_lens'] 
                distance_lens_to_aperture = form.cleaned_data['distance_lens_to_aperture'] * 1e-2 
                focal_length = form.cleaned_data['focal_length'] * 1e-3 

                distance_screen_to_aperture = form.cleaned_data['distance_screen_to_aperture'] * 1e-2 
                screen_width = form.cleaned_data['screen_width'] * 1e-3 
                screen_height = screen_width 

                resolution = form.cleaned_data['resolution']   
                animation_frames = form.cleaned_data['animation_frames']  
                animation_framerate = form.cleaned_data['animation_framerate']  
                x = np.linspace(-screen_width / 2, screen_width / 2, resolution)
                y = np.linspace(-screen_height / 2, screen_height / 2, resolution)
                xv, yv = np.meshgrid(x, y)

                if aperture_type == 'N-Slit':
                    slit_positions = np.linspace(
                        -(number_of_slits - 1) * distance_between_slits / 2,
                        (number_of_slits - 1) * distance_between_slits / 2,
                        number_of_slits
                    )
                    U0 = np.zeros_like(xv)
                    for position in slit_positions:
                        U0 += (np.abs(xv - position) < slit_width / 2) & (np.abs(yv) < slit_height / 2)

                elif aperture_type == "Circular":
                   
                    U0 = (xv**2 + yv**2 <= aperture_radius**2).astype(float)
                elif aperture_type == "QWT":
                    image_path = 'static/fourieroptics/QWT.png'
                    
                    
                    U0 = ApertureFromImage(image_path, xv.shape[0], yv.shape[1])
                

                def compute_U(U0, xv, yv, lam, z):
       
                    Nx, Ny = U0.shape

                    # Sampling intervals in x and y
                    dx = xv[0, 1] - xv[0, 0]  # dx
                    dy = yv[1, 0] - yv[0, 0]  # dy

                    # Spatial frequency coordinates in x and y (kx and ky)
                    kx = 2 * np.pi * fftfreq(Nx, dx)  # Frequency coordinates in x
                    ky = 2 * np.pi * fftfreq(Ny, dy)  # Frequency coordinates in y
                    
                    # 2D frequency grid (kx, ky)
                    kxv, kyv = np.meshgrid(kx, ky)
                    
                    # Wavenumber
                    k = 2 * np.pi / lam  # Wavenumber
                    
                    # Fourier transform of the initial field U0
                    A = fft2(U0)

                    # If lens effect is enabled
                    if add_lens:
                        # If the propagation is before the lens
                        if z <= distance_lens_to_aperture:
                            # Propagation using the transfer function in the Fourier domain
                            transfer_function = np.exp(1j * z * np.sqrt(k**2 - kxv**2 - kyv**2))
                            U = ifft2(A * transfer_function)  # Propagate using the transfer function
                            return U
                        
                        else:
                            # Propagation after the lens, split the distance into before and after lens
                            z_after_lens = z - distance_lens_to_aperture  # Distance after the lens
                            
                            # Apply lens phase shift (quadratic phase factor for the lens effect)
                            lens_phase = np.exp(-1j * (k / (2 * focal_length)) * (xv**2 + yv**2))  # Lens phase factor
                            U0_with_lens = U0 * lens_phase  # Apply lens phase to the field

                            # Fourier transform after applying lens phase shift
                            A_lens = fft2(U0_with_lens)
                            
                            # Transfer function for propagation before the lens (Fresnel approximation)
                            transfer_function_before_lens = np.exp(1j * distance_lens_to_aperture * np.sqrt(k**2 - kxv**2 - kyv**2))
                            A_after_lens = A_lens * transfer_function_before_lens  # Apply transfer function before lens

                            # Transfer function for propagation after the lens
                            transfer_function_after_lens = np.exp(1j * z_after_lens * np.sqrt(k**2 - kxv**2 - kyv**2))  # Fresnel transfer function after lens
                            U = ifft2(A_after_lens * transfer_function_after_lens)  # Propagate the field using the transfer function
                            
                            return U
                    else:
                        # If no lens, just propagate using the transfer function (Fresnel approximation)
                        transfer_function = np.exp(1j * z * np.sqrt(k**2 - kxv**2 - kyv**2))
                        U = ifft2(A * transfer_function)  # Propagate the field using the transfer function
                        return U

                
                rgb_images = []
                intensity_images = []
                frames_heatmap = []
                frames_3d = []
                frames_lines = []
                frames_rgb = []
                
                num_frames = animation_frames # 10
                rgb_colors = [wavelength_to_rgb(wl * 1e9) for wl in wavelengths]
                # Compute images for all frames
                
                for frame in range(num_frames):
                    screen_distance = distance_screen_to_aperture * frame / (num_frames - 1)

                    # Initialize images
                    rgb_image = np.zeros((*xv.shape, 3), dtype=np.float64)
                    intensity_image = np.zeros_like(xv, dtype=np.float64)
                        # Convert to nm for RGB mapping
                    U_total = np.zeros_like(xv, dtype=np.complex128)
                    for wl, intensity, rgb_color in zip(wavelengths, intensities, rgb_colors):
                        # Compute the field at the screen for the current wavelength
                        U_screen_temp = compute_U(U0, xv, yv, wl, screen_distance)

                        U_total += U_screen_temp * intensity

                        # Accumulate intensity and RGB contributions
                        intensity_image += np.abs(U_screen_temp) ** 2 * intensity
                        rgb_image[:, :, 0] += np.abs(U_screen_temp) ** 2 * intensity * rgb_color[0]
                        rgb_image[:, :, 1] += np.abs(U_screen_temp) ** 2 * intensity * rgb_color[1]
                        rgb_image[:, :, 2] += np.abs(U_screen_temp) ** 2 * intensity * rgb_color[2]

                    

                    intensity_distribution = np.abs(U_total) ** 2
                    intensity_distribution /= np.max(intensity_distribution)

                    # Normalize RGB image
                    max_value = np.max(rgb_image)
                    if max_value > 0:  # Only normalize if max value is greater than 0
                        rgb_image /= max_value
                    

                    # Store precomputed images
                    intensity_images.append(intensity_distribution)
                    rgb_images.append(rgb_image)

                    frame_data_heatmap = go.Heatmap(z=intensity_distribution, x=xv[0] * 1e3, y=yv[:, 0] * 1e3, colorscale='Inferno')
                    frames_heatmap.append(go.Frame(data=[frame_data_heatmap], name=str(frame)))

                    # 3D surface frame
                    frame_data_3d = go.Surface(z=intensity_distribution, x=xv[0] * 1e3, y=yv[:, 0] * 1e3, colorscale='Inferno')
                    frames_3d.append(go.Frame(data=[frame_data_3d], name=str(frame)))

                    # Line plots
                    intensity_y0 = intensity_distribution[len(y)//2, :]

                    frame_data_lines = [
                        go.Scatter(x=x * 1e3, y=intensity_y0, mode='lines', name="y = 0"),
                    ]
                    frames_lines.append(go.Frame(data=frame_data_lines, name=str(frame)))
                    
                    rgb_image = np.flip(rgb_image, axis=0)
                    
                    fig_rgb = px.imshow(
                        rgb_image,
                        x=xv[0] * 1e3,  # Convert to mm
                        y=yv[:, 0] * 1e3,  # Convert to mm
                        origin='lower',
                        aspect='auto',
                    )
                    fig_rgb.update_layout(
                        title='RGB Intensity Distribution at the Screen Plane',
                        xaxis=dict(
                            title="X-Position [mm]",
                            scaleanchor="y"  # Ensure the x-axis and y-axis have the same scale
                        ),
                        yaxis=dict(
                            title="Y-Position [mm]",
                            autorange='reversed'  # Reverse the y-axis
                        ),
                        width=600,
                        height=600,
                    )
                    
                    frames_rgb.append(go.Frame(data=fig_rgb.data, name=str(frame)))


                fig_rgb, ax_rgb = plt.subplots(figsize=(6, 6))
                ax_rgb.set_xlabel("x (mm)")
                ax_rgb.set_ylabel("y (mm)")
                ax_rgb.set_xlim(-screen_width * 1e3 / 2, screen_width * 1e3 / 2)
                ax_rgb.set_ylim(-screen_height * 1e3 / 2, screen_height * 1e3 / 2)
                img_rgb = ax_rgb.imshow(np.zeros((resolution, resolution, 3)), origin='lower',
                                        extent=[-screen_width * 1e3 / 2, screen_width * 1e3 / 2, -screen_height * 1e3 / 2, screen_height * 1e3 / 2])

                # Update function for RGB animation
                def update_rgb(frame):
                    img_rgb.set_data(rgb_images[frame])
                    ax_rgb.set_title(f"RGB Animation - Distance: {distance_screen_to_aperture * frame / (num_frames - 1) * 1e2:.2f} cm")
                    return [img_rgb]

                # Animation setup for Intensity
                fig_intensity, ax_intensity = plt.subplots(figsize=(6, 6))
                ax_intensity.set_xlabel("x (mm)")
                ax_intensity.set_ylabel("y (mm)")
                ax_intensity.set_xlim(-screen_width * 1e3 / 2, screen_width * 1e3 / 2)
                ax_intensity.set_ylim(-screen_height * 1e3 / 2, screen_height * 1e3 / 2)
                img_intensity = ax_intensity.imshow(np.zeros_like(intensity_images[0]), origin='lower', cmap='inferno',
                                                    extent=[-screen_width * 1e3 / 2, screen_width * 1e3 / 2,
                                                            -screen_height * 1e3 / 2, screen_height * 1e3 / 2],
                                                    vmin=0, vmax=1)

                # Update function for Intensity animation
                def update_intensity(frame):
                    img_intensity.set_data(intensity_images[frame])
                    ax_intensity.set_title(f"Intensity Distribution - Distance: {distance_screen_to_aperture * frame / (num_frames - 1) * 1e2:.2f} cm")
                    return [img_intensity]

                # Animation parameters
                framerate = animation_framerate
                interval = 1000 / framerate

                # Create and save the RGB animation
                ani_rgb = FuncAnimation(fig_rgb, update_rgb, frames=num_frames, interval=interval, blit=True)
                gif_file_path_rgb = 'static/fourieroptics/rgb_animation_optimized.gif'
                writer = PillowWriter(fps=framerate)
                ani_rgb.save(gif_file_path_rgb, writer=writer)
                shutil.copy(gif_file_path_rgb, 'staticfiles/fourieroptics/rgb_animation_optimized.gif')

                # Create and save the Intensity animation
                ani_intensity = FuncAnimation(fig_intensity, update_intensity, frames=num_frames, interval=interval, blit=True)
                gif_file_path_intensity = 'static/fourieroptics/intensity_animation_optimized.gif'
                ani_intensity.save(gif_file_path_intensity, writer=writer)
                shutil.copy(gif_file_path_intensity, 'staticfiles/fourieroptics/intensity_animation_optimized.gif')


                
                num_frames = animation_frames
                screen_distances = np.linspace(0, distance_screen_to_aperture, num_frames)

                slider_steps = [
                    {
                        "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": f"{screen_distance * 100:.2f} cm",  # Convert to cm
                        "method": "animate"
                    } for i, screen_distance in enumerate(screen_distances)
                ]

                # Heatmap animation
                fig_heatmap = go.Figure(
                    data=frames_heatmap[0].data,
                    layout=go.Layout(
                        title="Intensity Distribution at the Screen Plane",
                        xaxis=dict(
                            title="X-Position [mm]",
                            scaleanchor="y"  # Ensure the x-axis and y-axis have the same scale
                        ),
                        yaxis=dict(
                            title="Y-Position [mm]"
                        ),
                        width=600,
                        height=600,
                        sliders=[{
                            "currentvalue": {
                                "prefix": "Screen Distance: ",
                                "font": {"size": 20}
                            },
                            "steps": slider_steps,
                            "transition": {"duration": 300},
                            "x": 0.1,
                            "y": -0.1,  # Position the slider below the x-axis
                            "len": 1
                        }]
                    ),
                    frames=frames_heatmap
                )
                anim_heatmap_plot_data = pio.to_json(fig_heatmap)

                # RGB animation
                fig_rgb = go.Figure(
                    data=frames_rgb[0].data,
                    layout=go.Layout(
                        title="RGB Intensity Distribution at the Screen Plane",
                        xaxis=dict(
                            title="X-Position [mm]",
                            scaleanchor="y"  # Ensure the x-axis and y-axis have the same scale
                        ),
                        yaxis=dict(
                            title="Y-Position [mm]"
                        ),
                        width=600,
                        height=600,
                        sliders=[{
                            "currentvalue": {
                                "prefix": "Screen Distance: ",
                                "font": {"size": 20}
                            },
                            "steps": slider_steps,
                            "transition": {"duration": 300},
                            "x": 0.1,
                            "y": -0.1,  # Position the slider below the x-axis
                            "len": 1
                        }]
                    ),
                    frames=frames_rgb
                )
                anim_rgb_plot_data = pio.to_json(fig_rgb)

                # 3D surface plot animation
                fig_3d = go.Figure(
                    data=frames_3d[0].data,
                    layout=go.Layout(
                        title="3D Surface Plot of Intensity at the Screen Plane",
                        scene=dict(
                            xaxis_title="X-Position [mm]",
                            yaxis_title="Y-Position [mm]",
                            zaxis_title="Intensity"
                        ),
                        width=600,
                        height=600,
                        sliders=[{
                            "currentvalue": {
                                "prefix": "Screen Distance: ",
                                "font": {"size": 20}
                            },
                            "steps": slider_steps,
                            "transition": {"duration": 300},
                            "x": 0.1,
                            "y": -0.1,  # Position the slider below the x-axis
                            "len": 1
                        }]
                    ),
                    frames=frames_3d
                )
                anim_3d_plot_data = pio.to_json(fig_3d)

                # Line plots animation
                fig_lines = go.Figure(
                    data=frames_lines[0].data,
                    layout=go.Layout(
                        title="Intensity Distributions Along Y=0",
                        xaxis_title="Position [mm]",
                        yaxis_title="Intensity",
                        width=600,
                        height=600,
                        sliders=[{
                            "currentvalue": {
                                "prefix": "Screen Distance: ",
                                "font": {"size": 20}
                            },
                            "steps": slider_steps,
                            "transition": {"duration": 300},
                            "x": 0.1,
                            "y": -0.1,  # Position the slider below the x-axis
                            "len": 1
                        }]
                    ),
                    frames=frames_lines
                )
                anim_lines_plot_data = pio.to_json(fig_lines)

                if "plot_all_optimized" in request.POST:
                    context = {
                        'form': form,
                        'formset': formset,
                        'anim_heatmap_plot_data': anim_heatmap_plot_data,
                        'anim_3d_plot_data': anim_3d_plot_data,
                        'anim_lines_plot_data': anim_lines_plot_data,
                        'anim_rgb_plot_data': anim_rgb_plot_data,
                        'animation_saved': True, 
                    }
                    return render(request, 'fourieroptics/home.html', context)

                elif "save_simu" in request.POST:
                    title = request.POST.get('title')
                    visibility = request.POST.get('visibility')
                    simulation_result = FourierOptics.objects.create(
                        title = title,
                        visibility=visibility,
                        created_by=request.user,
                        aperture_type=aperture_type,
                        wavelengths=wavelengths,
                        intensities=intensities,
                        number_of_slits=number_of_slits,
                        distance_between_slits=distance_between_slits,
                        slit_width=slit_width,
                        slit_height=slit_height,
                        aperture_radius=aperture_radius,
                        add_lens=add_lens,
                        distance_lens_to_aperture=distance_lens_to_aperture,
                        focal_length=focal_length,
                        distance_screen_to_aperture=distance_screen_to_aperture,
                        screen_width=screen_width,
                        screen_height=screen_height,
                        resolution=resolution,
                        animation_frames=animation_frames,
                        animation_framerate=animation_framerate,
                        anim_heatmap_plot_data=anim_heatmap_plot_data,
                        anim_3d_plot_data = anim_3d_plot_data,
                        anim_lines_plot_data = anim_lines_plot_data,
                        anim_rgb_plot_data = anim_rgb_plot_data
                    )

                    # Define the unique paths for the GIF files
                    gif_file_path_rgb_sim = f'fourieroptics/rgb_animation_{simulation_result.id}_optimized.gif'
                    gif_file_path_intensity_sim = f'fourieroptics/intensity_animation_{simulation_result.id}_optimized.gif'

                    # Copy the GIF files to the static directory (relative path)
                    shutil.copy(gif_file_path_rgb, f'static/{gif_file_path_rgb_sim}')
                    shutil.copy(gif_file_path_intensity, f'static/{gif_file_path_intensity_sim}')

                    # Copy the GIFs to the staticfiles directory (useful for production deployment)
                    shutil.copy(f'static/{gif_file_path_rgb_sim}', f'staticfiles/{gif_file_path_rgb_sim}')
                    shutil.copy(f'static/{gif_file_path_intensity_sim}', f'staticfiles/{gif_file_path_intensity_sim}')

                    # Save the relative paths to the model
                    simulation_result.rgb_animation_path = gif_file_path_rgb_sim
                    simulation_result.intensity_animation_path = gif_file_path_intensity_sim

                    # Save the simulation result instance
                    simulation_result.save()
                   
                    return redirect(simulation_result.get_absolute_url())


                    #return redirect('app:view_simulation', simulation_id=simulation_result.id,user_id=request.user.id)

    else:
        form = FourierDiff()
        WavelengthIntensityFormSet = formset_factory(WavelengthIntensityForm, extra=1)
        formset = WavelengthIntensityFormSet()
        
        context = {'form': form,'formset':formset}

    return render(request, 'fourieroptics/home.html', context)
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.http import JsonResponse

import plotly.express as px

def wavelength_to_rgb(wavelength):
    gamma = 0.8
    intensity_max = 255
    factor = 0.0
    r = g = b = 0

    if (wavelength >= 380) and (wavelength < 440):
        r = -(wavelength - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif (wavelength >= 440) and (wavelength < 490):
        r = 0.0
        g = (wavelength - 440) / (490 - 440)
        b = 1.0
    elif (wavelength >= 490) and (wavelength < 510):
        r = 0.0
        g = 1.0
        b = -(wavelength - 510) / (510 - 490)
    elif (wavelength >= 510) and (wavelength < 580):
        r = (wavelength - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif (wavelength >= 580) and (wavelength < 645):
        r = 1.0
        g = -(wavelength - 645) / (645 - 580)
        b = 0.0
    elif (wavelength >= 645) and (wavelength <= 750):
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        r = g = b = 0.0

    if (wavelength >= 380) and (wavelength < 420):
        factor = 0.3 + 0.7*(wavelength - 380) / (420 - 380)
    elif (wavelength >= 420) and (wavelength < 645):
        factor = 1.0
    elif (wavelength >= 645) and (wavelength <= 750):
        factor = 0.3 + 0.7*(750 - wavelength) / (750 - 645)
    else:
        factor = 0.0

    if r != 0:
        r = round(intensity_max * ((r * factor) ** gamma))
    if g != 0:
        g = round(intensity_max * ((g * factor) ** gamma))
    if b != 0:
        b = round(intensity_max * ((b * factor) ** gamma))

    return (r, g, b)
