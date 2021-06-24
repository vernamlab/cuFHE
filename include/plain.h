#pragma once


#include <iostream>
#include <random>
#include <vector>
using namespace std;

namespace cufhe {
void NotCheck(uint8_t& out, const uint8_t& in) { out = (~in) & 0x1; }

void CopyCheck(uint8_t& out, const uint8_t& in) { out = in; }

void NandCheck(uint8_t& out, const uint8_t& in0, const uint8_t& in1)
{
    out = 1 - in0 * in1;
}

void OrCheck(uint8_t& out, const uint8_t& in0, const uint8_t& in1)
{
    out = (in0 + in1) > 0;
}

void OrYNCheck(uint8_t& out, const uint8_t& in0, const uint8_t& in1)
{
    out = (in0 + (1 - in1)) > 0;
}

void OrNYCheck(uint8_t& out, const uint8_t& in0, const uint8_t& in1)
{
    out = ((1 - in0) + in1) > 0;
}

void AndCheck(uint8_t& out, const uint8_t& in0, const uint8_t& in1)
{
    out = in0 * in1;
}

void AndYNCheck(uint8_t& out, const uint8_t& in0, const uint8_t& in1)
{
    out = in0 * (1 - in1);
}

void AndNYCheck(uint8_t& out, const uint8_t& in0, const uint8_t& in1)
{
    out = (1 - in0) * in1;
}

void XorCheck(uint8_t& out, const uint8_t& in0, const uint8_t& in1)
{
    out = (in0 + in1) & 0x1;
}

void XnorCheck(uint8_t& out, const uint8_t& in0, const uint8_t& in1)
{
    out = (~(in0 ^ in1)) & 0x1;
}

void MuxCheck(uint8_t& out, const uint8_t& inc, const uint8_t& in1,
              const uint8_t& in0)
{
    out = inc ? in1 : in0;
}
}
