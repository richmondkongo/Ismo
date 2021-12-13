from selenium import webdriver
from time import sleep, time
import numpy as np
import pandas as pd

url = r'https://m.facebook.com/societegenerale.cotedivoire/posts/?ref=page_internal&mt_nav=0'
browser = webdriver.Chrome('browser/chromedriver.exe')
#browser = webdriver.PhantomJS('browser/phantomjs.exe')
browser.get(url)
sleep(2)
browser.execute_script("window.scrollTo(0, window.scrollY + 10000)")
sleep(2)
posts = browser.find_elements_by_css_selector('a._5msj')
links = []
df_post = pd.DataFrame(columns=['id', 'contenu_post', 'img_post', 'nb_commentaires_post', 'lien_post', 'nb_commentaire_pos', 'nb_commentaire_neg'])
df_commentaire = pd.DataFrame(columns=['id_post', 'contenu_post', 'img_post', 'contenu_commentaire', 'auteur_commentaire', 'pp_auteur', 'prediction'])

for p in posts:
    # obtention de tout les liens vers les publications
    links.append(p.get_attribute("href"))

for l in links:
    # on parcours la liste des liens obtenus afi d'en extraire le contenu (post, commentaire, image)
    browser.get(l)
    sleep(1)
    post = 'div.story_body_container > div._5rgt._5nk5 > div'
    contenu_post = browser.find_elements_by_css_selector(post)[0].get_attribute('innerHTML')
    img = 'div.story_body_container > div._5rgu._7dc9._27x0 > div._5uso._5t8z > a > div > div > img'
    sleep(1)
    img_post = browser.find_elements_by_css_selector(img)[0].get_attribute('src')

    # on va à présent chercher tout les commentaires du post donné
    post = 'div#m_story_permalink_view'
    post = browser.find_elements_by_css_selector(post)[0]
    comment = ' div._2b04 > div._14v5 > div > div:nth-child(2)'
    auteur = ' div._2b04 > div._14v5 > div > div._2b05'
    img_auteur = ' div._2a_j > div > div:nth-child(1) > img'
    auteurs = post.find_elements_by_css_selector(auteur)
    pp_auteurs = post.find_elements_by_css_selector(img_auteur)
    nb_comments = 0
    classe_pos = 0
    classe_neg = 0
    id_post = str(time()).split('.')
    id_post = ''.join(id_post)

    for i, c in enumerate(post.find_elements_by_css_selector(comment)):
        #comments.append({'commentaire':c.get_attribute('innerHTML'), 'auteur': auteurs[i].get_attribute("innerHTML"), 'pp_auteur': pp_auteurs[i].get_attribute('src')})
        nb_comments += 1
        classe = np.random.randint(2, size=1)[0]
        if classe == 1:
            classe_pos += 1
        else:
            classe_neg += 1
        df_commentaire = df_commentaire.append({ 'id_post': id_post, 'contenu_post': contenu_post, 'img_post': img_post, 'contenu_commentaire':c.get_attribute('innerHTML'), 'auteur_commentaire': auteurs[i].get_attribute("innerHTML"), 'pp_auteur': pp_auteurs[i].get_attribute('src'), 'prediction': classe }, ignore_index=True)
    
    # on ajoute les données obtenues dans notre dataframe
    df_post = df_post.append({ 'id': id_post, 'contenu_post': contenu_post, 'img_post': img_post, 'nb_commentaires_post': nb_comments, 'lien_post': l, 'nb_commentaire_pos': classe_pos, 'nb_commentaire_neg': classe_neg}, ignore_index=True)



# oon sauvegarde notre dataframe
df_commentaire.to_hdf('data/commentaire.h5', key='df_commentaire', mode='w')
df_post.to_hdf('data/post.h5', key='df_commentaire', mode='w')
#df_commentaire.to_csv('data.csv', sep=',')

browser.close()
browser.quit()

"""
file1 = open("myfile.txt","wb")
file1.write(str(browser.page_source).encode('utf8'))
file1.close() #to change file access modes
"""

