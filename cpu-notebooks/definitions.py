pipa_path = '/nfs/juhu/data/rakhasan/pipa-dataset/'
openImg_path = '/nfs/juhu/data/rakhasan/bystander-detection/google-img-db/'
survey_path='/nfs/juhu/data/rakhasan/bystander-detection/pilot-study2/'
survey_photo_path = survey_path+'/photos/'

img_level_concepts = ['person_size','person_distance_axes_norm','num_people']
high_level_concepts = ['was_aware', 'posing','comfort', 'will', 'photographer_intention', 'replacable', 'photo_place']
high_level_concepts_num = [c+'_num' for c in high_level_concepts]

high_level_concepts_name = {'was_aware_num':'Awareness', 'posing_num':'Pose',
                            'comfort_num':'Comfort', 'will_num':'Willingness', 
                            'photographer_intention_num':'Photographer intention',
                            'replacable_num':'Replaceable', 'photo_place_num':'Photo place',
                            'person_distance_axes_norm':'Distance',
                            'person_size':'Size', 'num_people':'Number of people'}


#joint names labeled by openpose
body_joint_names = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb',
               'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 
               'Leye', 'Reye', 'Lear', 'Rear']

#angles between pairs of body joint, from openpose
link_angle_features = ['angle_'+str(i) for i in range(17)]

#probability of detecting a body joint, from openpose
body_joint_prob_features = [j + '_prob' for j in body_joint_names]

face_exp_feaures = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

visual_features = img_level_concepts +\
    link_angle_features + body_joint_prob_features + face_exp_feaures
