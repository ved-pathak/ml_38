# Greeting Card Generator 
 # Overview of Problem Statement
 Problem statement is to Develop a pipeline for creating greeting and wishing cards for users. The system should generate personalized cards that incorporate details such as the occasion, recipient’s name, and design preferences specified by the user.
 # Model Used
Stable Diffusion 3 Medium is a Multimodal Diffusion Transformer (MMDiT) text-to-image model that boasts significant enhancements in image quality, typography, complex prompt     comprehension, and resource efficiency.

The Phi-3 family models have been refined through supervised learning and preference adjustments to enhance their ability to follow instructions and ensure safety. They excel in various benchmarks, including common sense, language skills, mathematics, coding, long-term context, and logical reasoning, especially when compared to other models with fewer than 13 billion parameters.
 # Approach
We have used two approach for this Problem

  1. We used Phi-3 to refine and expand the user's prompt into a detailed description, which is then fed into the Stable Diffusion model. This process ensures that Stable Diffusion generates the most relevant image for creating the greeting card according to the user's specifications.
  
  2. We used the Stable Diffusion model to create the background image for the greeting card, Phi-3 to generate a relevant message based on the provided prompt, and Pillow to combine the text with the image, producing the final card.
  
# Final Implementation
We combined both approaches to achieve the best possible results for the user. By leveraging both methods, we generated four different greeting cards. This dual approach allowed us to refine and enhance the output, ensuring that the final cards were highly relevant and personalized to meet the user's needs.

# How To Run 
1. Creating and Activating a Virtual Environment:

                          python -m venv venv
                          source venv/bin/activate  # For macOS/Linux
                          venv\Scripts\activate     # For Windows

                         
 2. Installing Packages:

                          pip install -r requirements.txt
    
 3. Run File


                          python app.py



   

                                           

# How To Use Model

1. Enter your prompt for the greeting card.
2. You will receive four different greeting card options. Select the one you prefer and download it.
