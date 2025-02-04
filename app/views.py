from django.shortcuts import render, get_object_or_404 
import os
import yaml

from django.http import JsonResponse
import os

from django.core.mail import EmailMessage

from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate,logout
from django.shortcuts import redirect

from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode
from django.utils.http import urlsafe_base64_decode
from django.utils.encoding import force_bytes
from django.utils.encoding import force_str

from django.template import loader
from django.core.mail import EmailMessage
from django.core.mail import send_mail
import random
from django.shortcuts import render
from django.http import Http404
from .models import FourierOptics , Profile
from django.contrib.auth.models import User
from django.http import HttpResponse
def contact(request):
    
    
    if request.method == 'POST':
        # Process the form data if the request method is POST
        name = request.POST.get('name', '')
        email = request.POST.get('email', '')
        subject = request.POST.get('subject', '')
        message = request.POST.get('message', '')
      
        if name!="" and email != "" and message!="":
            send_mail(
                'Subject: OptiqsHub New Contact Form Submission',
                f'Name: {name}\nsubject: {subject}\nEmail: {email}\nMessage: {message}',
                'optiqshub@gmail.com',  # Sender's email
                ["bakadir.oussama@gmail.com"],  # List of recipient emails
                fail_silently=False,
            )
            return HttpResponse('Thank you for your message! We will get back to you soon.')
    
    return render(request, 'app/contact.html')

def view_simulation(request, username, title):
    simulation_result = get_object_or_404(FourierOptics, created_by__profile__slug=username, title__iexact=title)
    
    return render(request, 'app/fourieroptics.html', 
                      {'result': simulation_result,
                    'lightsource':zip(simulation_result.wavelengths,simulation_result.intensities)})

def account_view(request, slug):
    profile = get_object_or_404(Profile, slug=slug)
    user = profile.user
    simulations = FourierOptics.objects.filter(created_by=user)
    context = {
        'user': user,
       'simulations': simulations,
    }
    return render(request, 'app/account.html', context)


def logout_view(request):
	logout(request)
	return redirect("app:home")

def forgot_password(request):
    if request.method == "POST":
        email = request.POST.get("email")
        user = User.objects.filter(email=email).first()
        if user:
            # Generate password reset token
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = default_token_generator.make_token(user)
            
            # Construct the password reset link (change 'reset_link' to your actual reset URL)
            reset_link = f"https://optiqshub.com/reset_password/{uid}/{token}/"
            subject = 'Optiqs Hub Reset Your Password'
            message = f"""
            <html>
                <body>
                <p>Hey {user.username},</p>
                <p>You can reset your password by clicking the link below:</p>
                <p><a href="{reset_link}" >Reset Your Password</a></p>
                <br>
                <p>Best regards,</p>
                <p>The Bjerseys Team</p>
            </body>
            </html>
            """

            send_mail(
            subject,
            '',  # Leave the plain text part empty as we're using HTML email
            'optiqshub@gmail.com',
            [email],
            html_message=message  # Pass the HTML message here
            )
            
            return render(request,'app/forgot_password.html',{'email_sent': True,'password_reset':True})
        else:
            # Handle case where email is not found in the database
            return render(request,'app/forgot_password.html', {'message': 'Email not found','password_reset':True})

    return render(request, 'app/forgot_password.html')

def signup(request):
    if request.method == "POST":

        if "signup" in request.POST:
            username_signup = request.POST["username_signup"]
            email_signup = request.POST["email_signup"]
            password_signup = request.POST["password_signup"]
            if User.objects.filter(username=username_signup).exists(): 
                context = {'message': "Username already exists. Please use a different username.","signup":True}
                return render(request, 'app/signup.html', context)
            
            elif User.objects.filter(email=email_signup).exists(): 
                context = {'message': "Email already exists. Please use a different email.","signup":True}
                return render(request, 'app/signup.html', context)
            else:
                user = User.objects.create_user(username=username_signup, email=email_signup, password=password_signup)
                profile, created = Profile.objects.get_or_create(user=user)
                profile.save()

                
            user = authenticate(request, username=username_signup, email=email_signup, password=password_signup)
            login(request, user)

            return redirect('app:home')
    

    return render(request, 'app/signup.html')

def login_view(request):
    if request.method == "POST":
        if "login" in request.POST:
            email_username = request.POST["email_username"]
            password = request.POST["password"]
            try:
                try:
                    user = User.objects.get(email=email_username)
                    email = user.email
                    username = user.username
                except:
                    user = User.objects.get(username=email_username)
                    email = user.email
                    username = user.username
            except User.DoesNotExist:
                username = None
            

            try:
                user = authenticate(request, username=username, email=email, password=password)
                if user is not None:
                    login(request, user)
                    return redirect('app:home')
            except:
                context = {'message': 'Invalid credentials'}
                return render(request, 'app/login.html', context)


    return render(request, 'app/login.html')
# Create your views here.
def home(request):
    # email_content = """
    #         Hello,

    #         We are excited to invite you to Optiqs Hub!

    #         Discover a world of possibilities, connect with like-minded individuals, and stay updated with the latest in our community.

    #         We look forward to having you onboard.

    #         Best regards,  
    #         The Optiqs Hub Team  
    #         """

    # email_subject = 'Optiqs Hub!'
    # from_email = 'optiqshub@gmail.com'
    # recipients = ["bakadir.oussama@gmail.com"]
    # for recipient in recipients:
    #     msg_me = EmailMessage(email_subject,email_content,from_email,[recipient])
    #     msg_me.content_subtype = 'html'  # Indicate that the email content is HTML
    #     try:
        
    #         msg_me.send()
    #         # Rest of your code after sending the email
    #     except Exception as e:
    #         print("Email sending error:", e)
    
    
    context = {
      
    }
    
    return render(request, 'app/home.html', context)

def get_categories_and_materials(base_path):
    categories = {}
    try:
        for category in os.listdir(base_path):
            category_path = os.path.join(base_path, category)
            if os.path.isdir(category_path):
                materials = {}
                for material in os.listdir(category_path):
                    material_path = os.path.join(category_path, material)
                    if os.path.isdir(material_path):
                        nk_files = []
                        nk_path = os.path.join(material_path, "nk")
                        if os.path.exists(nk_path):
                            nk_files = [
                                f for f in os.listdir(nk_path) if f.endswith(".yml")
                            ]
                        materials[material] = nk_files
                if materials:  # Only add to categories if materials are found
                    categories[category] = materials
    except Exception as e:
        print(f"An error occurred: {e}")
    return categories

def refractiveindex_data(request):
    base_path = "static/thinfilms"
    categories_and_materials = get_categories_and_materials(base_path)

    context = {
        'categories_and_materials': categories_and_materials,  # Pass the data to the template
        'base_url':base_path,
    }
    
    return render(request, 'app/refractiveindex_data.html', context)




