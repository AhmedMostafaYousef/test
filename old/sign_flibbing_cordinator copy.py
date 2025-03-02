from __future__ import print_function

import random
import statistics
import numpy as np
from numpy.linalg import norm
from collections import OrderedDict
import torch
from copy import deepcopy
from utils import utils

class SignFlibbingMaliciousCordinator():
    signFlippingAttackerBeforeAttackValue = {}
    normalClientDummyValue = {}
    signFlippingAttackers = []
    numberOfClientsTraining = 0
    numberOfSignFlippingAttackers = 0
    model = None
    minimumNumberOfMaliciousClientsThreshold = 0
    samplesThreshold = 5

    def addMaliciousClientUpdateOld(client):
        SignFlibbingMaliciousCordinator.signFlippingAttackers.append(client.cid)
        SignFlibbingMaliciousCordinator.numberOfSignFlippingAttackers += 1

        clientGradient = client.avg_delta

        SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackValue[client.cid] = deepcopy(clientGradient)

        result = {
            "newMaliciousClient": None,
            "isNewMalicoiusClientCalculated": False,
        }

        modelData = SignFlibbingMaliciousCordinator.model.state_dict()
        trainable_parameter = utils.getTrainableParameters(SignFlibbingMaliciousCordinator.model)

        # Check if all malicoius nodes before attack is ready to Calcualte malicoius nodes attack state
        if len(SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackValue) == SignFlibbingMaliciousCordinator.numberOfSignFlippingAttackers and SignFlibbingMaliciousCordinator.numberOfSignFlippingAttackers>=SignFlibbingMaliciousCordinator.minimumNumberOfMaliciousClientsThreshold:
            # Calcualte malicoius nodes attack state
            # print(len(SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackValue))
            # print(SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackValue.keys())
            # print()
            listOfMaliciousClientsIdsToChooseFrom = list(SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackValue.keys())
            
            # Randomly assign value from the malicious before nodes to each normal client
            for i in range(SignFlibbingMaliciousCordinator.numberOfClientsTraining):
                if i not in SignFlibbingMaliciousCordinator.signFlippingAttackers:
                    selectedRandomMaliciousKey = random.sample(listOfMaliciousClientsIdsToChooseFrom, k=1)[0]
                    SignFlibbingMaliciousCordinator.normalClientDummyValue[i] = (SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackValue[selectedRandomMaliciousKey])

            listOfParamters = {}
            listOfParamtersMean = {}
            listOfParamtersStandardDivation = {}
            newMaliciousClient = {}

            # TODO stable point 1

            listOfNewMaliciousClient = {}
            listOfNewMaliciousClientNumberOfClientsMarkedAsMalicious = {}


            for t in range(SignFlibbingMaliciousCordinator.samplesThreshold):
                for p in SignFlibbingMaliciousCordinator.model.state_dict():
                    if p not in trainable_parameter:
                        continue
                    listOfParamters[p] = []
                    listOfParamtersMean[p] = []
                    listOfParamtersStandardDivation[p] = []

                    for i in SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackValue:
                        malicoiusClient = (SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackValue[i])
                        listOfParamters[p].append(malicoiusClient[p])

                    arr = np.array(listOfParamters[p])
                    listOfParamtersMean[p] = np.mean(arr, axis=0)
                    listOfParamtersStandardDivation[p] = np.std(arr, axis=0)

                    A = np.array(modelData[p])
                    B = np.array(listOfParamtersMean[p])

                    if len(list(A.shape))==0:
                        A = np.array([modelData[p]])
                        B = np.array([listOfParamtersMean[p]])

                    cosine = np.sum(A*B)/(norm(A)*norm(B))

                    flattened_cosine = cosine.flatten()
                    mean_flattened_cosine = flattened_cosine.mean()

                    numLow = np.array(listOfParamtersMean[p]) - (4 * np.array(listOfParamtersStandardDivation[p]))
                    numHigh = np.array(listOfParamtersMean[p]) - (3 * np.array(listOfParamtersStandardDivation[p]))

                    if mean_flattened_cosine < 0:
                        numLow = np.array(listOfParamtersMean[p]) + (3 * np.array(listOfParamtersStandardDivation[p]))
                        numHigh = np.array(listOfParamtersMean[p]) + (4 * np.array(listOfParamtersStandardDivation[p]))

                    newSelectedSampledValues = np.full_like(numLow, 0)

                    for index, value in np.ndenumerate(numLow):
                        rangeStart = value
                        rangeEnd = numHigh[index]

                        difference = rangeEnd - rangeStart
                        step = difference / SignFlibbingMaliciousCordinator.samplesThreshold

                        if step !=0 and (not np.isnan(step)):
                            tempRange = np.arange(rangeStart, rangeEnd, step)
                            sampledRangeOfValues = list(tempRange)
                            flip = np.random.choice([-1, -1])
                            # new_value = sampledRangeOfValues[t] * flip
                            # new_value = value * flip
                            new_value = listOfParamters[p][len(SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackValue)-1][index] * flip
                            newSelectedSampledValues[index] = new_value
                        else:
                            newSelectedSampledValues[index] = value

                    newMaliciousClient[p] = torch.tensor(newSelectedSampledValues)

                # STAGE 1: Collect short HoGs for bengin clients
                short_HoGs = {}
                for k in range(SignFlibbingMaliciousCordinator.numberOfClientsTraining):
                    if k not in SignFlibbingMaliciousCordinator.signFlippingAttackers:
                        # shortHoGs
                        temp = SignFlibbingMaliciousCordinator.normalClientDummyValue[k]
                        short_HoGs[k] = torch.cat([v.flatten() for v in temp.values()]).detach().cpu().numpy()

                keys = SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackValue.keys()
                key = list(keys)[SignFlibbingMaliciousCordinator.numberOfSignFlippingAttackers-1]
                temp = newMaliciousClient
                short_HoGs[key] = torch.cat([v.flatten() for v in temp.values()]).detach().cpu().numpy()

                # STAGE 2 - STEP 1: Detect FLIP_SIGN gradient attackers
                flip_sign_id = set()
                non_mal_sHoGs = dict(short_HoGs)
                median_sHoG = np.median(np.array(list(non_mal_sHoGs.values())), axis=0)
                for i, v in short_HoGs.items():
                    v = np.array(list(v))
                    d_cos = np.dot(median_sHoG, v)/(np.linalg.norm(median_sHoG)*np.linalg.norm(v))
                    if d_cos < 0: # angle > 90
                        if i != key:
                            flip_sign_id.add(i)
                    else:
                        if i == key:
                            flip_sign_id.add(i)

                listOfNewMaliciousClientNumberOfClientsMarkedAsMalicious[t] = len(flip_sign_id)
                listOfNewMaliciousClient[t] = OrderedDict(newMaliciousClient)

            
            items = list(listOfNewMaliciousClientNumberOfClientsMarkedAsMalicious.items())
            larget_value_index = 0
            for i in range(len(items)):
                larget_value_index = i
                for j in range(i + 1, len(items)):
                    if items[j][1] >= items[larget_value_index][1]:  # Compare values
                        larget_value_index = j

            # for i in listOfNewMaliciousClientNumberOfClientsMarkedAsMalicious:
            #     print("threshold", i)
            #     print("total detected", listOfNewMaliciousClientNumberOfClientsMarkedAsMalicious[i])
            #     print()

            result["newMaliciousClient"] = listOfNewMaliciousClient[larget_value_index]
            result["isNewMalicoiusClientCalculated"] = True

            # TODO stable point 2

            # for p in SignFlibbingMaliciousCordinator.model.state_dict():

            #     if p not in trainable_parameter:
            #         continue
            #     listOfParamters[p] = []
            #     listOfParamtersMean[p] = []
            #     listOfParamtersStandardDivation[p] = []

            #     for i in SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackValue:
            #         malicoiusClient = SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackValue[i]
            #         listOfParamters[p].append(malicoiusClient[p])

            #     arr = np.array(listOfParamters[p])
            #     listOfParamtersMean[p] = np.mean(arr, axis=0)
            #     listOfParamtersStandardDivation[p] = np.std(arr, axis=0)

            #     A = np.array(modelData[p])
            #     B = np.array(listOfParamtersMean[p])

            #     if len(list(A.shape))==0:
            #         A = np.array([modelData[p]])
            #         B = np.array([listOfParamtersMean[p]])

            #     cosine = np.sum(A*B)/(norm(A)*norm(B))

            #     flattened_cosine = cosine.flatten()
            #     mean_flattened_cosine = flattened_cosine.mean()

            #     numLow = np.array(listOfParamtersMean[p]) - (4 * np.array(listOfParamtersStandardDivation[p]))
            #     numHigh = np.array(listOfParamtersMean[p]) - (3 * np.array(listOfParamtersStandardDivation[p]))

            #     if mean_flattened_cosine < 0:
            #         numLow = np.array(listOfParamtersMean[p]) + (3 * np.array(listOfParamtersStandardDivation[p]))
            #         numHigh = np.array(listOfParamtersMean[p]) + (4 * np.array(listOfParamtersStandardDivation[p]))

            #     newSelectedSampledValues = np.full_like(numLow, 0)

            #     for index, value in np.ndenumerate(numLow):
            #         rangeStart = value
            #         rangeEnd = numHigh[index]

            #         difference = rangeEnd - rangeStart
            #         step = difference / SignFlibbingMaliciousCordinator.samplesThreshold

            #         if step !=0 and (not np.isnan(step)):
            #             tempRange = np.arange(rangeStart, rangeEnd, step)
            #             sampledRangeOfValues = list(tempRange)
            #             random.shuffle(sampledRangeOfValues)
            #             flip = np.random.choice([-1, -1])
            #             # new_value = sampledRangeOfValues[0] * flip
            #             # new_value = listOfParamters[p][SignFlibbingMaliciousCordinator.numberOfSignFlippingAttackers -1][index] * flip
            #             new_value = value * flip
            #             # new_value = numHigh[index] * flip
            #             newSelectedSampledValues[index] = new_value
            #         else:
            #             newSelectedSampledValues[index] = value

            #     # choosenMalicoiusClientParamterValue = listOfParamters[p][SignFlibbingMaliciousCordinator.numberOfSignFlippingAttackers -1]
            #     newMaliciousClient[p] = torch.tensor(newSelectedSampledValues)
            
            # result["newMaliciousClient"] = OrderedDict(newMaliciousClient)
            # result["isNewMalicoiusClientCalculated"] = True

            # TODO stable point 2
        return result
