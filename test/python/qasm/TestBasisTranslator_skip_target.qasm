OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.07822540000000) q[2];
cx q[2],q[1];
rz(-2.07822540000000) q[1];
cx q[2],q[1];
rz(2.07822540000000) q[1];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[3],q[2];
rz(-pi/4) q[2];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[3],q[2];
rz(pi/4) q[2];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[2];
rz(-pi/4) q[2];
rz(pi/4) q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[3],q[2];
rz(3*pi/4) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[3],q[4];
rz(pi/4) q[3];
rz(-pi/4) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[2],q[1];
rz(0.486224710000000) q[1];
cx q[2],q[1];
rz(-pi) q[2];
sx q[2];
rz(3*pi/4) q[2];
cx q[1],q[2];
rz(pi/4) q[1];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
rz(pi/4) q[2];
sx q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[3],q[4];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
rz(-pi/4) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
rz(pi/4) q[4];
cx q[3],q[4];
rz(pi/4) q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
rz(-pi/4) q[4];
cx q[3],q[4];
cx q[3],q[2];
rz(-pi/4) q[2];
rz(pi/4) q[3];
cx q[3],q[2];
rz(-3*pi/4) q[3];
sx q[3];
rz(-pi/2) q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
rz(0.9066446) q[3];
rz(3*pi/4) q[4];
sx q[4];
rz(-2.4791672) q[4];
cx q[4],q[3];
rz(0.90837085) q[3];
sx q[3];
rz(-1.8522083) q[3];
sx q[3];
cx q[4],q[3];
sx q[3];
rz(-1.8522083) q[3];
sx q[3];
rz(-2.6004136) q[3];
cx q[3],q[2];
rz(-pi/4) q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[0],q[1];
rz(pi/4) q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(-pi/4) q[1];
cx q[0],q[1];
rz(3*pi/4) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[0],q[1];
rz(pi/4) q[0];
rz(-pi/4) q[1];
cx q[0],q[1];
rz(2.6059615) q[0];
sx q[0];
rz(-1.7104962) q[0];
sx q[0];
rz(-0.49205251) q[0];
cx q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[2],q[1];
rz(-pi/4) q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
rz(-pi/2) q[4];
sx q[4];
rz(-pi/2) q[4];
cx q[3],q[4];
cx q[3],q[2];
rz(pi/4) q[2];
cx q[1],q[2];
rz(pi/4) q[1];
rz(-pi/4) q[2];
cx q[3],q[2];
rz(3*pi/4) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[2],q[1];
rz(-pi/4) q[1];
rz(pi/4) q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
rz(-2.499918) q[1];
cx q[3],q[2];
rz(-0.78925375) q[2];
cx q[1],q[2];
rz(-0.6416747) q[2];
sx q[2];
rz(-1.7578332) q[2];
sx q[2];
cx q[1],q[2];
sx q[2];
rz(-1.7578332) q[2];
sx q[2];
rz(1.4309284) q[2];
rz(1.264262) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[3];
