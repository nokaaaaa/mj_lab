#pragma once

#include <iostream>
#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include "joystick/joystick.h"
#include <memory>
#include <GLFW/glfw3.h>

// Set from main() once the MuJoCo viewer window exists (before any joystick
// object is constructed), so KeyboardJoystick can poll key state from it.
inline GLFWwindow* g_glfw_window = nullptr;


class XBoxJoystick : public unitree::common::UnitreeJoystick
{
public:
    XBoxJoystick(std::string device, int bits = 15)
	: unitree::common::UnitreeJoystick()
	{
		js_ = std::make_unique<Joystick>(device);
		if(!js_->isFound()) {
			std::cout << "Error: Joystick open failed." << std::endl;
			exit(1);
		}
        max_value_ = 1 << (bits - 1);
	}

    void update() override
    {
        js_->getState();
        back(js_->button_[6]);
        start(js_->button_[7]);
        LB(js_->button_[4]);
        RB(js_->button_[5]);
        A(js_->button_[0]);
        B(js_->button_[1]); 
        X(js_->button_[2]);
        Y(js_->button_[3]);
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);
        LT(js_->axis_[2] > 0);
        RT(js_->axis_[5] > 0);
        lx(double(js_->axis_[0]) / max_value_);
        ly(-double(js_->axis_[1]) / max_value_);
        rx(double(js_->axis_[3]) / max_value_);
        ry(-double(js_->axis_[4]) / max_value_);
    }
private:
	std::unique_ptr<Joystick> js_;
	int max_value_;
};


// Emulates a Unitree WirelessController from the MuJoCo viewer window's
// keyboard, for when no physical gamepad is available.
//
// Mapping (held-down = pressed):
//   F                  -> F1   (bind to a single-key "FixStand" transition)
//   M                  -> F2   (bind to a single-key "Mimic" transition)
//   S                  -> back (bind to a single-key "emergency stop" transition)
//   Arrow keys         -> D-pad (up/down/left/right)
//   A / B / X / Y      -> A / B / X / Y
//   Left/Right Shift   -> LT / RT
//   Left/Right Ctrl    -> LB / RB
//   Enter              -> start
class KeyboardJoystick : public unitree::common::UnitreeJoystick
{
public:
    explicit KeyboardJoystick(GLFWwindow* window) : window_(window)
    {
        if (!window_) {
            std::cout << "Error: Keyboard joystick has no window to read from." << std::endl;
            exit(1);
        }
    }

    void update() override
    {
        auto held = [this](int key) { return glfwGetKey(window_, key) == GLFW_PRESS; };

        F1(held(GLFW_KEY_F) ? 1 : 0);
        F2(held(GLFW_KEY_M) ? 1 : 0);
        back(held(GLFW_KEY_S) ? 1 : 0);
        start(held(GLFW_KEY_ENTER) ? 1 : 0);
        LB(held(GLFW_KEY_LEFT_CONTROL) ? 1 : 0);
        RB(held(GLFW_KEY_RIGHT_CONTROL) ? 1 : 0);
        A(held(GLFW_KEY_A) ? 1 : 0);
        B(held(GLFW_KEY_B) ? 1 : 0);
        X(held(GLFW_KEY_X) ? 1 : 0);
        Y(held(GLFW_KEY_Y) ? 1 : 0);
        up(held(GLFW_KEY_UP) ? 1 : 0);
        down(held(GLFW_KEY_DOWN) ? 1 : 0);
        left(held(GLFW_KEY_LEFT) ? 1 : 0);
        right(held(GLFW_KEY_RIGHT) ? 1 : 0);
        LT(held(GLFW_KEY_LEFT_SHIFT) ? 1.0f : 0.0f);
        RT(held(GLFW_KEY_RIGHT_SHIFT) ? 1.0f : 0.0f);
        lx(0.0f);
        ly(0.0f);
        rx(0.0f);
        ry(0.0f);
    }
private:
    GLFWwindow* window_;
};


class SwitchJoystick : public unitree::common::UnitreeJoystick
{
public:
    SwitchJoystick(std::string device, int bits = 15)
	: unitree::common::UnitreeJoystick()
	{
		js_ = std::make_unique<Joystick>(device);
		if(!js_->isFound()) {
			std::cout << "Error: Joystick open failed." << std::endl;
			exit(1);
		}
        max_value_ = 1 << (bits - 1);
	}

    void update() override
    {
        js_->getState();
        back(js_->button_[10]);
        start(js_->button_[11]);
        LB(js_->button_[6]);
        RB(js_->button_[7]);
        A(js_->button_[0]);
        B(js_->button_[1]); 
        X(js_->button_[3]);
        Y(js_->button_[4]);
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);
        LT(js_->axis_[5] > 0);
        RT(js_->axis_[4] > 0);
        lx(double(js_->axis_[0]) / max_value_);
        ly(-double(js_->axis_[1]) / max_value_);
        rx(double(js_->axis_[2]) / max_value_);
        ry(-double(js_->axis_[3]) / max_value_);
    }
private:
	std::unique_ptr<Joystick> js_;
	int max_value_;
};