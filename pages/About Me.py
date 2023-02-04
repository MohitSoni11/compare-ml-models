#############
## Imports ##
#############

import streamlit as st

#########################
## Important Variables ##
#########################

about_me = '''
<img style='float: left; border-radius: 10%; margin-right: 15px; width: 35%;' src='https://media.licdn.com/dms/image/C4D03AQGhoDOyhz1MEg/profile-displayphoto-shrink_400_400/0/1658108717556?e=1680739200&v=beta&t=Zx3OSlBPY05-0os4l3Lbjv6QM6sX_zIdaHLrJ3P_2s4'>

Hi there! I'm Mohit Soni, a high school student with a passion for both computer science and
sports. I've completed two 3-month long internships -- where I was the ML & Web-Developer Intern -- and I've
also participated in a Machine Learning research project with a University of Washington professor. I'm skilled
in a variety of programming languages including Python, C++, Java, HTML, CSS, JavaScript, etc. When I'm not
studying/coding, you can find me on the cricket field as a national player. I'm very competitive and am always
looking for new challenges and opportunities, so feel free to reach out!'''

inspiration = '''
As a passionate machine learning enthusiast, I had already acquired a solid understanding of
the fundamental concepts in this vast field through completion of multiple courses on Coursera.
However, I aspired to put my knowledge into practical use by building real-world products. In my
search for the right platform, I came across a YouTube video that introduced me to Streamlit. The
video highlighted the simplicity and versatility of Streamlit in creating machine learning applications,
and I was immediately intrigued.

I was impressed by the ease of use and the intuitive interface of Streamlit, which allowed me to
focus on the development of my machine learning models without getting bogged down by the complexities
of the underlying technology. The video demonstrated how Streamlit could be used to quickly create
interactive data visualizations, dashboard-style reports, and other interactive applications for 
machine learning. This was exactly what I was looking for to bring my projects to life. I was eager to
dive in and start using Streamlit for my own projects.'''

contact = '''
<center style=''>
<a href='https://www.linkedin.com/in/mohit-soni-b55a18225/'>LinkedIn</a>
<a href='https://github.com/MohitSoni11' style='margin-left: 5%;'>Github</a>
</center>
'''

#################
## Application ##
#################

st.title('About Me')
st.write(about_me, unsafe_allow_html=True)

st.header('Inspiration')
st.write(inspiration, unsafe_allow_html=True)

st.write(contact, unsafe_allow_html=True)

