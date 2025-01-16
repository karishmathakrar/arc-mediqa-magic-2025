import os
import sys

import json
import yaml

import urllib
import praw

#########
# pip install pyyaml
# pip install praw

# def download_image( url, post_id, ftype, output_dir ) :
#     print( url )
#     response = urllib.request.urlopen( url )
#     img = response.read()

#     outfn = '{}/{}.{}'.format( output_dir, str( post_id ), ftype )
#     if os.path.exists( outfn ) :
#         print('SKIPPING - {}'.format(outfn))
#         return

#     with open( outfn,'wb') as f:
#         f.write(img)

import time

def download_image(url, post_id, ftype, output_dir):
    print(url)
    try:
        response = urllib.request.urlopen(url)
        img = response.read()

        outfn = '{}/{}.{}'.format(output_dir, str(post_id), ftype)
        if os.path.exists(outfn):
            print('SKIPPING - {}'.format(outfn))
            return

        with open(outfn, 'wb') as f:
            f.write(img)
    except urllib.error.HTTPError as e:
        print(f"HTTPError for {url}: {e}")
        return
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return

    time.sleep(1)  # Add a 1-second delay


if __name__ == '__main__':

    if len( sys.argv ) < 5 :
        print('Usage: python download_data.py <config-yaml-file> <input-file> <output-file> <image-dir-output>')
        sys.exit()

    fn_yaml = sys.argv[1]
    input_file =  sys.argv[2]
    output_file =  sys.argv[3]
    img_output_dir =  sys.argv[4]

    print( 'Reading credentials from {}'.format(fn_yaml) )
    print( 'Reading input from {}'.format(input_file) )

    if os.path.exists( output_file ) :
        print( 'Output file {} exists, we shall output to COPY-{}'.format( output_file,output_file ) )
        output_file = 'COPY-' + output_file
    
    print( 'Printing output to {}'.format(output_file) )
    print( 'Saving images to output directory {}'.format(img_output_dir) )

    os.makedirs( img_output_dir, exist_ok=True )
    
    with open( fn_yaml, 'r') as f :
        config = yaml.safe_load( f )
    
    reddit = praw.Reddit(
            client_id= config['CLIENT_ID'],
            client_secret=config['SECRET_TOKEN'],
            user_agent="my user agent",
        )
    
    with open( input_file, encoding='utf-8' ) as f :
        json_obj = json.load( f )

        for ind, item in enumerate( json_obj ) :
            post_id = item['encounter_id']
            post = reddit.submission( id=post_id )

            print( 'working on {} - {}'.format( ind, post_id ) )

            item['query_title_en'] = post.title
            item['query_content_en'] = post.selftext

            url = post.url
            ftype = url[-3:]

            if url == '' :
                print('IMAGE URL NOT EXISTING!!')
                continue

            download_image( url, post_id, ftype, img_output_dir )

    # with open( output_file, 'w' ) as f :
    #     json.dump( json_obj, f, ensure_ascii=False, indent=4 )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=4)

