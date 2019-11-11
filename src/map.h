#ifndef SRC_MAP_H_
#define SRC_MAP_H_

class Map{
public:
  Map();
  ~Map() {}

protected:
  //  TODO
  
  long unsigned int mnMaxKFid;
  
  // Index related to a big change in the map (loop closure, global BA)
  int mnBigChangeIdx;

};

#endif // SRC_MAP_H_