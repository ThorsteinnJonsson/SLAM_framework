#include "keyframe_database.h"

#include "DBoW2/BowVector.h"

KeyframeDatabase::KeyframeDatabase (const std::shared_ptr<OrbVocabulary>& voc) 
      : mpVoc(voc) {
  mvInvertedFile.resize(voc->size());
}

void KeyframeDatabase::add(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(mMutex);

  for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(); 
                                        vit != pKF->mBowVec.end(); 
                                        ++vit) {
    mvInvertedFile[vit->first].push_back(pKF);
  }
}

void KeyframeDatabase::erase(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(mMutex);

  // Erase elements in the Inverse File for the entry
  for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(); 
                                        vit != pKF->mBowVec.end();
                                        vit++) 
  {
    // List of keyframes that share the word
    std::list<KeyFrame*>& lKFs = mvInvertedFile[vit->first];
    for(std::list<KeyFrame*>::iterator lit = lKFs.begin(); 
                                  lit != lKFs.end(); 
                                  ++lit) 
    {
      if (pKF == *lit) {
        lKFs.erase(lit);
        break;
      }
    }
  }
}

void KeyframeDatabase::Clear() {
  mvInvertedFile.clear();
  mvInvertedFile.resize(mpVoc->size());
}


std::vector<KeyFrame*> KeyframeDatabase::DetectLoopCandidates(KeyFrame* pKF, 
                                                              float minScore) {
  std::set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
  std::list<KeyFrame*> lKFsSharingWords;

  // Search all keyframes that share a word with current keyframes
  // Discard keyframes connected to the query keyframe
  {
    std::unique_lock<std::mutex> lock(mMutex);

    for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(); 
                                          vit != pKF->mBowVec.end(); 
                                          ++vit)
    {
      std::list<KeyFrame*>& lKFs = mvInvertedFile[vit->first];

      for (std::list<KeyFrame*>::iterator lit = lKFs.begin(); 
                                          lit != lKFs.end(); 
                                          ++lit)
      {
        KeyFrame* pKFi = *lit;
        if (pKFi->mnLoopQuery != pKF->Id()) {
          pKFi->mnLoopWords = 0;
          if (!spConnectedKeyFrames.count(pKFi)) {
            pKFi->mnLoopQuery = pKF->Id();
            lKFsSharingWords.push_back(pKFi);
          }
        }
        ++pKFi->mnLoopWords;
      }
    }
  }

  if (lKFsSharingWords.empty()) {
    return std::vector<KeyFrame*>();
  }

  std::list<std::pair<float,KeyFrame*>> lScoreAndMatch;

  // Only compare against those keyframes that share enough words
  int maxCommonWords = 0;
  for (std::list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(); //std algorithm replace possible?
                                      lit != lKFsSharingWords.end(); 
                                      ++lit)
  {
    if ((*lit)->mnLoopWords > maxCommonWords) {
      maxCommonWords = (*lit)->mnLoopWords;
    }
  }

  const int minCommonWords = static_cast<int>(maxCommonWords * 0.8f);

  // Compute similarity score. Retain the matches whose score is higher than minScore
  int nscores = 0;
  for (std::list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(); 
                                 lit != lKFsSharingWords.end(); 
                                 ++lit) {
    KeyFrame* pKFi = *lit;
    if (pKFi->mnLoopWords > minCommonWords) {
      ++nscores;
      pKFi->mLoopScore = mpVoc->score(pKF->mBowVec,
                                      pKFi->mBowVec);
      if (pKFi->mLoopScore >= minScore) {
        lScoreAndMatch.push_back(std::make_pair(pKFi->mLoopScore, pKFi));
      }
    }
  }

  if (lScoreAndMatch.empty()) {
    return std::vector<KeyFrame*>();
  }

  // Lets now accumulate score by covisibility
  float bestAccScore = minScore;
  std::list<std::pair<float,KeyFrame*>> lAccScoreAndMatch;
  for (auto it = lScoreAndMatch.begin(); 
            it != lScoreAndMatch.end(); 
            ++it)
  {
    KeyFrame* pKFi = it->second;
    
    float bestScore = it->first;
    float accScore = it->first;
    KeyFrame* pBestKF = pKFi;

    std::vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);
    for (auto vit = vpNeighs.begin(); 
              vit != vpNeighs.end(); 
              ++vit)
    {
      KeyFrame* pKF2 = *vit;
      if (pKF2->mnLoopQuery == pKF->Id() 
          && pKF2->mnLoopWords > minCommonWords) {
        accScore += pKF2->mLoopScore;
        if (pKF2->mLoopScore > bestScore) {
          pBestKF = pKF2;
          bestScore = pKF2->mLoopScore;
        }
      }
    }

    lAccScoreAndMatch.push_back(std::make_pair(accScore, pBestKF));
    if (accScore > bestAccScore) {
      bestAccScore = accScore;
    }
  }

  // Return all those keyframes with a score higher than 0.75*bestScore
  const float minScoreToRetain = 0.75f * bestAccScore;

  std::set<KeyFrame*> spAlreadyAddedKF;
  std::vector<KeyFrame*> vpLoopCandidates;
  vpLoopCandidates.reserve(lAccScoreAndMatch.size());

  for (auto it = lAccScoreAndMatch.begin(); 
            it != lAccScoreAndMatch.end(); 
            ++it)
  {
    if (it->first > minScoreToRetain) {
      KeyFrame* pKFi = it->second;
      if (!spAlreadyAddedKF.count(pKFi)) {
        vpLoopCandidates.push_back(pKFi);
        spAlreadyAddedKF.insert(pKFi);
      }
    }
  }

  return vpLoopCandidates;
}


std::vector<KeyFrame*> KeyframeDatabase::DetectRelocalizationCandidates(const Frame& frame) {
  std::list<KeyFrame*> lKFsSharingWords;

  // Search all keyframes that share a word with current frame
  {
    std::unique_lock<std::mutex> lock(mMutex);

    for (DBoW2::BowVector::const_iterator vit = frame.GetBowVector().begin(); 
                                          vit != frame.GetBowVector().end(); 
                                          ++vit) 
    {
      std::list<KeyFrame*>& lKFs = mvInvertedFile[vit->first];
      for (auto lit = lKFs.begin(); 
                lit != lKFs.end(); 
                ++lit)
      {
        KeyFrame* pKFi = *lit;
        if (pKFi->mnRelocQuery != frame.Id()) {
          pKFi->mnRelocWords = 0;
          pKFi->mnRelocQuery = frame.Id();
          lKFsSharingWords.push_back(pKFi);
        }
        ++pKFi->mnRelocWords;
      }
    }
  }

  if(lKFsSharingWords.empty()) {
    return std::vector<KeyFrame*>();
  }
      

  // Only compare against those keyframes that share enough words
  int maxCommonWords = 0;
  for (auto lit = lKFsSharingWords.begin(); 
            lit != lKFsSharingWords.end(); 
            ++lit)
  {
    if ((*lit)->mnRelocWords > maxCommonWords) {
      maxCommonWords = (*lit)->mnRelocWords;
    }
  }

  const int minCommonWords = static_cast<int>(maxCommonWords * 0.8f);

  // Compute similarity score.
  int nscores = 0;
  std::list<std::pair<float,KeyFrame*>> lScoreAndMatch;
  for (auto lit = lKFsSharingWords.begin(); 
            lit != lKFsSharingWords.end(); 
            ++lit)
  {
    KeyFrame* pKFi = *lit;
    if (pKFi->mnRelocWords > minCommonWords) {
      ++nscores;
      pKFi->mRelocScore = mpVoc->score(frame.GetBowVector(),
                                       pKFi->mBowVec);
      lScoreAndMatch.push_back(std::make_pair(pKFi->mRelocScore, pKFi));
    }
  }

  if(lScoreAndMatch.empty()) {
    return std::vector<KeyFrame*>();
  }

  float bestAccScore = 0;
  std::list<std::pair<float,KeyFrame*>> lAccScoreAndMatch;
  
  // Accumulate score by covisibility
  for (auto it = lScoreAndMatch.begin(); 
            it != lScoreAndMatch.end(); 
            ++it)
  {
    KeyFrame* pKFi = it->second;
    
    float bestScore = it->first;
    float accScore = bestScore;
    KeyFrame* pBestKF = pKFi;

    std::vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);
    for (auto vit = vpNeighs.begin(); 
              vit != vpNeighs.end(); 
              ++vit)
    {
      KeyFrame* pKF2 = *vit;
      if (pKF2->mnRelocQuery != frame.Id()) {
        continue;
      }

      accScore += pKF2->mRelocScore;
      if (pKF2->mRelocScore > bestScore) {
        pBestKF = pKF2;
        bestScore = pKF2->mRelocScore;
      }
    }
    lAccScoreAndMatch.push_back(std::make_pair(accScore, pBestKF));
    if (accScore > bestAccScore) {
      bestAccScore = accScore;
    }
  }

  // Return all those keyframes with a score higher than 0.75*bestScore
  const float minScoreToRetain = 0.75f * bestAccScore;
  std::set<KeyFrame*> spAlreadyAddedKF;
  std::vector<KeyFrame*> vpRelocCandidates;
  vpRelocCandidates.reserve(lAccScoreAndMatch.size());

  for (auto it = lAccScoreAndMatch.begin(); 
            it != lAccScoreAndMatch.end(); 
            ++it)
  {
    if (it->first > minScoreToRetain) {
      KeyFrame* pKFi = it->second;
      if (!spAlreadyAddedKF.count(pKFi)) {
        vpRelocCandidates.push_back(pKFi);
        spAlreadyAddedKF.insert(pKFi);
      }
    }
  }
  return vpRelocCandidates;
}
