//
//  ValStructVec.h
//  ImageTracking
//
//  Created by bittnt on 29/11/2014.
//  Copyright (c) 2014 Xin Sun. All rights reserved.
//

#ifndef ImageTracking_ValStructVec_h
#define ImageTracking_ValStructVec_h
#include "kyheader.h"
/************************************************************************/
/* A value struct vector that supports efficient sorting                */
/************************************************************************/

template<typename VT, typename ST>
struct ValStructVec
{
    ValStructVec(){clear();}
    inline int size() const {return sz;}
    inline void clear() {sz = 0; structVals.clear(); valIdxes.clear();}
    inline void reserve(int resSz){clear(); structVals.reserve(resSz); valIdxes.reserve(resSz); }
    inline void pushBack(const VT& val, const ST& structVal) {valIdxes.push_back(std::make_pair(val, sz)); structVals.push_back(structVal); sz++;}
    
    inline const VT& operator ()(int i) const {return valIdxes[i].first;} // Should be called after sort
    inline const ST& operator [](int i) const {return structVals[valIdxes[i].second];} // Should be called after sort
    inline VT& operator ()(int i) {return valIdxes[i].first;} // Should be called after sort
    inline ST& operator [](int i) {return structVals[valIdxes[i].second];} // Should be called after sort
    
    void sort(bool descendOrder = true);
    const std::vector<ST> &getSortedStructVal();
    void append(const ValStructVec<VT, ST> &newVals, int startV = 0);
    
    std::vector<ST> structVals; // struct values
    
private:
    int sz; // size of the value struct vector
    std::vector<std::pair<VT, int>> valIdxes; // Indexes after sort
    bool smaller() {return true;};
    std::vector<ST> sortedStructVals;
};

template<typename VT, typename ST>
void ValStructVec<VT, ST>::append(const ValStructVec<VT, ST> &newVals, int startV)
{
    int sz = newVals.size();
    for (int i = 0; i < sz; i++)
        pushBack((float)((i+300)*startV)/*newVals(i)*/, newVals[i]);
}

template<typename VT, typename ST>
void ValStructVec<VT, ST>::sort(bool descendOrder /* = true */)
{
    if (descendOrder)
        std::sort(valIdxes.begin(), valIdxes.end(), std::greater<std::pair<VT, int>>());
    else
        std::sort(valIdxes.begin(), valIdxes.end(), std::less<std::pair<VT, int>>());
}

template<typename VT, typename ST>
const std::vector<ST>& ValStructVec<VT, ST>::getSortedStructVal()
{
    sortedStructVals.resize(sz);
    for (int i = 0; i < sz; i++)
        sortedStructVals[i] = structVals[valIdxes[i].second];
    return sortedStructVals;
}

/*
 void valStructVecDemo()
 {
	ValStructVec<int, string> sVals;
	sVals.pushBack(3, "String 3");
	sVals.pushBack(5, "String 5");
	sVals.pushBack(4, "String 4");
	sVals.pushBack(1, "String 1");
	sVals.sort(false);
	for (int i = 0; i < sVals.size(); i++)
 printf("%d, %s\n", sVals(i), _S(sVals[i]));
 }
 */

#endif
