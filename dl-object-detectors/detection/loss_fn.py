import torch
import torch.nn as nn
import numpy as np

# private
import detection_utils

def cross_entropy_loss(outputs, labels):
    # Apply log softmax on outputs
    log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
    
    # Create a tensor with the probabilities of the correct classes for each sample
    actual_log_probs = log_probs.gather(1, labels.view(-1, 1)).squeeze()

    # Negate since we want to minimize the negative log likelihood
    loss = -actual_log_probs

    return loss


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = detection_utils.cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        device = predicted_locs.device
        boxes  = [boxes[i].to(device) for i in range(len(boxes))]
        labels = [labels[i].to(device) for i in range(len(boxes))]

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)


        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = detection_utils.find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)
            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = detection_utils.cxcy_to_gcxgcy(detection_utils.xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)


        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)
        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = cross_entropy_loss(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / (n_positives.sum().float() + 1e-8)  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss







class MultiBoxLoss_Mine(torch.nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super().__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = detection_utils.cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        # self.smooth_l1 = torch.nn.L1Loss()
        self.smooth_l1 = torch.nn.SmoothL1Loss()
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none') # NOTE : why reduce = False?


    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
            1. predicted_locs : (B, 8732, 4)
                - predicted locations (boxes) with respect to the 8732 prior boxes (offset)
            2. predicted_scores : (B, 8732, n_classes)
                - scores
            3. boxes  : list of B tensors in shape of (n_obj, 4)
            4. labels : list of B tensors in shape of (n_obj,)
        """
        device = predicted_locs.device
        boxes  = [boxes[i].to(device) for i in range(len(boxes))]
        labels = [labels[i].to(device) for i in range(len(boxes))]

        B = predicted_locs.size(0)
        P = self.priors_cxcy.size(0) # num_priors
        C = predicted_scores.size(2) # num_classes

        assert P == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((B, P, 4), dtype=torch.float).to(device) # (B, 8732, 4)
        true_classes = torch.zeros((B, P), dtype=torch.long).to(device) # (B, 8732)

        # for each image
        for i in range(B):
            num_objects = boxes[i].size(0)
            
            overlap = detection_utils.find_jaccard_overlap(boxes[i], self.priors_xy) # (num_objects, 8732)

            # for each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0) # (8732)

            """
            we don't want a situation where an object is not represented in our positive (non-background) priors.
            
            1. an object might not be the best object for all priors, and is therefore not in 'object_for_each_prior'. 
            (e.g., predicted:car | label : cat)
            
            2. all priors with the object may be assigned as background based on the threshold (0.5).
            (e.g., all priors' label : background)
            """
            
            # To remedy this,
            # first, find the prior that has the maximum overlap for each object.
            # 8732개의 prior 중 object와 가장 많이 겹치는 (box와 가장 많이 겹치는) 놈 찾기
            _, prior_for_each_object = overlap.max(dim=1) # (num_objects)

            # then, assign each object to the corresponding maximum-overlap-prior
            # 그리고 object가 여러개일 테니까 각 object target에 제일 많이 겹치는 prior들을 assign 함
            # (이 아래부터 이해가 잘 안 됨!!)
            # NOTE : this fixes problem 1.
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(num_objects)).to(device)

            # to ensure these priors qualify, artificially give them an overlap of greather than 0.5.
            # NOTE : this fixes problem 2.
            overlap_for_each_prior[prior_for_each_object] = 1.

            # labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior] # (8732)
            # set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0 # (8732)

            # store
            true_classes[i] = label_for_each_prior
            # encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = detection_utils.cxcy_to_gcxgcy(
                detection_utils.xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy) # (8732, 4)
            
        # identify priors that are positive (object, non-background)
        positive_priors = true_classes != 0 # (B, 8732)


        """
        Localization loss
        """
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors]) # (), scalar

        # note : indexing with a torch.uint8 (byte) tensor 
        # flattens the tensor when indexing is across multiple dimensions ( B & 8732 )
        # so, if predicted_locs has the shape (B, 8732, 4), 
        # predicted_locs[positive_priors] will have (total positives, 4)

        
        """
        Confidence loss
        - confidence loss is computed over positive priors and the mos difficult (hardest negative priors in each image)
        - that is, for each image,
        - we will take the "HARDEST" (neg_pos_ratio * n_positives) negative priors, i.e., where there is maximum loss
        - this is called hard negative mining - it concentrates on hardest negatives in each image,
        - and also minimizes pos/neg imbalance
        """

        # number of positive and har-negative priors per image
        n_positives = positive_priors.sum(dim=1) # (B)
        n_hard_negatives = self.neg_pos_ratio * n_positives # (B)

        # first, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, C), true_classes.view(-1)) # (B * 8732)
        conf_loss_all = conf_loss_all.view(B, P) # (B, 8732)

        # we already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors] # (sum(n_positives))

        # next, find which priors are hard-negative
        # to do this, sort only negative priors in each image in order of decreasing loss
        # and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone() # (B, 8732)
        conf_loss_neg[positive_priors] = 0. # positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True) # (B, 8732)
        
        hardness_ranks = torch.LongTensor(range(P)).unsqueeze(0).expand_as(conf_loss_neg).to(device) # (B, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsuqeeze(1) # (B, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives] # (sum (n_hard_negatives))

        # as in the paper, averaged over positive priors only, 
        # although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float() # scalar

        return conf_loss + self.alpha*loc_loss
        
        




            

            

            

