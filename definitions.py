pipa_path = '/nfs/juhu/data/rakhasan/pipa-dataset/'
openImg_path = '/nfs/juhu/data/rakhasan/bystander-detection/google-img-db/'
survey_path='/nfs/juhu/data/rakhasan/bystander-detection/pilot-study2/'
survey_photo_path = survey_path+'/photos/'

img_level_concepts = ['person_size','person_distance_axes_norm','number_people']
high_level_concepts = ['was_aware', 'posing','comfort', 'will', 'photographer_intention', 'replacable', 'photo_place']
high_level_concepts_num = [c+'_num' for c in high_level_concepts]

high_level_concepts_name = {'was_aware_num':'Awareness', 'posing_num':'Pose',
                            'comfort_num':'Comfort', 'will_num':'Willingness', 
                            'photographer_intention_num':'Photographer intention',
                            'replacable_num':'Replaceable', 'photo_place_num':'Photo place'}

